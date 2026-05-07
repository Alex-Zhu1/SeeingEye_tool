import base64
import asyncio
from typing import Dict, List, Optional, Union, Tuple, Any

from pydantic import Field

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.logger import logger
from app.utils.log_save import LogSave
from app.config import config

from app.prompt.translator import DIRECT_VISION_PROMPT
from app.prompt.text_only_reasoning import DIRECT_REASONING_PROMPT

class IterativeRefinementFlow(BaseFlow):
    """
    A flow that manages iterative refinement between translator and reasoning agents.
    
    The flow orchestrates:
    1. Translator generates initial SIR from image
    2. Reasoning agent analyzes SIR and either provides answer or feedback
    3. If feedback provided, translator refines SIR based on feedback
    4. Repeat until final answer or max iterations
    """
    
    max_iterations: int = Field(default_factory=lambda: config.flow_config.max_iterations)
    current_iteration: int = Field(default=0)
    base64_image: Optional[str] = Field(default=None)
    image_path: Optional[str] = Field(default=None)
    current_sir: Optional[str] = Field(default=None)
    previous_sir: Optional[str] = Field(default=None)  # SIR from previous iteration
    log_save: LogSave = Field(default_factory=LogSave)

    def __init__(
        self,
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]],
        **data
    ):
        super().__init__(agents, **data)

        # Validate required agents
        if "translator" not in self.agents:
            raise ValueError("IterativeRefinementFlow requires a 'translator' agent")
        if "reasoning" not in self.agents:
            raise ValueError("IterativeRefinementFlow requires a 'reasoning' agent")

    @property
    def translator_agent(self) -> BaseAgent:
        """Get the translator agent"""
        return self.agents["translator"]

    @property
    def reasoning_agent(self) -> BaseAgent:
        """Get the reasoning agent"""
        return self.agents["reasoning"]

    def _reset_agent_memory_for_iteration(self, agent: BaseAgent, iteration: int) -> None:
        """Reset agent memory but preserve SIR from previous iteration for context."""
        # Clear current memory
        agent.memory.clear()

        # For iterations > 1, provide access to previous SIR for context
        if iteration > 1 and self.previous_sir:
            if agent == self.translator_agent:
                context_msg = f"Your previous visual description (iteration {iteration-1}): {self.previous_sir}\n\nUse this as reference to improve your current description."
            else:  # reasoning agent
                context_msg = f"Previous visual description (iteration {iteration-1}): {self.previous_sir}\n\nThis shows the previous attempt at visual analysis. You can reference this for continuity."

            # Add system message directly to avoid base64_image parameter issue
            from app.schema import Message
            system_msg = Message.system_message(context_msg)
            agent.memory.add_message(system_msg)

    def _update_sir_history(self, new_sir: str) -> None:
        """Update SIR history - store current as previous, set new as current."""
        # Store current SIR as previous
        if self.current_sir:
            self.previous_sir = self.current_sir

        # Set new SIR as current
        self.current_sir = new_sir
        logger.info(f"📝Flow SIR updated based translator output SIR result")

    def _append_feedback_to_sir(self, feedback: str) -> None:
        import re

        if not feedback or not feedback.strip():
            logger.warning("⚠️ Attempted to append empty feedback to SIR")
            return
        if not self.current_sir:
            logger.warning("⚠️ No current SIR to append feedback to")
            return

        clean_feedback = feedback.strip()

        # 情况1：iter1 强制 CONTINUE 时，feedback 是 FINAL ANSWER 格式
        if "FINAL ANSWER:" in clean_feedback and "Reasoning:" in clean_feedback:
            answer_match = re.search(r'FINAL ANSWER:\s*(.+)', clean_feedback)
            reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\n\n|$)', clean_feedback, re.DOTALL)
            preliminary = answer_match.group(1).strip() if answer_match else "unclear"
            reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""
            clean_feedback = (
                f"Preliminary answer: {preliminary}\n"
                f"Reasoning basis: {reasoning_text}\n"
                f"Still need: Use smart_grid_caption to extract precise spatial layout, "
                f"exact labels, and numerical values needed to confirm the answer."
            )

        # 情况2：正常 feedback，剥离 agent loop 的输出包装
        else:
            # 剥离 "Step N: Observed output of cmd `terminate_and_ask_translator` executed:" 包装层
            tool_output_match = re.search(
                r'Observed output of cmd `terminate_and_ask_translator` executed:\s*(.+)',
                clean_feedback,
                re.DOTALL
            )
            if tool_output_match:
                clean_feedback = tool_output_match.group(1).strip()
                logger.info("✂️ Stripped agent loop wrapper from feedback")

            # 剥离 "feedback:" 前缀
            if clean_feedback.lower().startswith("feedback:"):
                clean_feedback = clean_feedback[len("feedback:"):].strip()

            # 剥离 "FEEDBACK from reasoning agent:" 前缀
            if clean_feedback.startswith("FEEDBACK from reasoning agent:"):
                clean_feedback = clean_feedback.replace("FEEDBACK from reasoning agent:", "").strip()

        # 清理旧的 REASONING FEEDBACK 段，每轮只保留最新一条
        if "--- REASONING FEEDBACK ---" in self.current_sir:
            self.current_sir = self.current_sir.split("--- REASONING FEEDBACK ---")[0].rstrip()
            logger.info("✂️ Removed previous REASONING FEEDBACK from SIR")

        self.current_sir = f"{self.current_sir}\n\n--- REASONING FEEDBACK ---\n{clean_feedback}"
        logger.info(f"💬 Appended feedback to SIR (new length: {len(self.current_sir)} characters)")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert an image file to base64 encoded string."""
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string

    def _parse_input(self, input_text: str, image_path: Optional[str] = None) -> Tuple[str, List[str], str]:
        """Parse input text to extract question, options, and image path."""
        lines = input_text.strip().split('\n')

        # Find question and options by splitting on "Options:"
        input_parts = input_text.split("Options:")
        if len(input_parts) >= 2:
            question = input_parts[0].strip()
            options_and_image = input_parts[1].strip()

            # Extract all option lines, not just the first one
            import re
            options_lines = []
            for line in options_and_image.split('\n'):
                line = line.strip()
                if line and not line.startswith("image_path:"):
                    # Check if this line looks like an option (starts with letter followed by period or parenthesis)
                    if re.match(r'^[A-J]\.', line) or re.match(r'^\([A-J]\)', line) or not options_lines:
                        options_lines.append(line)
                else:
                    # Stop at image_path or empty lines after we've found options
                    if options_lines:
                        break

            # Join all option lines and parse them
            options_text = '\n'.join(options_lines)
            options = self._parse_multiline_options(options_text)
        else:
            # 取全部有效行作为question，而不是只取第一行
            question = '\n'.join(
                line for line in lines 
                if line.strip() and not line.startswith("image_path:")
            ).strip()
            options = []

        # Extract image path from input or use provided parameter
        if image_path is None:
            for line in lines:
                if line.startswith("image_path:"):
                    image_path = line.split(":", 1)[1].strip()
                    break

        if not image_path:
            raise ValueError("No image path provided")

        return question, options, image_path

    def _parse_options(self, options_line: str) -> List[str]:
        """Parse options line handling various formats."""
        if not options_line or options_line.lower() in ['none', 'n/a', '']:
            return []

        try:
            if options_line.startswith('[') and options_line.endswith(']'):
                import ast
                return ast.literal_eval(options_line)
            else:
                # Smart parsing for financial amounts
                import re
                option_pattern = r',\s+(?=\$?[A-Za-z0-9])'
                parts = re.split(option_pattern, options_line)
                options = [opt.strip() for opt in parts if opt.strip()]

                if not options and options_line.strip():
                    options = [opt.strip() for opt in options_line.split(',') if opt.strip()]
                return options
        except:
            return [opt.strip() for opt in options_line.split(',') if opt.strip()]

    def _parse_multiline_options(self, options_text: str) -> List[str]:
        """Parse multiline options text, preserving the full formatted options."""
        if not options_text or options_text.lower() in ['none', 'n/a', '']:
            return []

        options = []
        import re

        # Split by newlines and process each line
        for line in options_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Check if line has a letter prefix (A., B., (A), (B), etc.)
            if re.match(r'^(?:\([A-J]\)|\b[A-J]\.)', line):
                # Keep the full formatted option with letter label
                options.append(line)
            elif line:
                # If no letter prefix found, assume it's a continuation or standalone option
                options.append(line)

        return options

    def _setup_image(self, image_path: str) -> None:
        """Load and encode image for processing."""
        self.image_path = image_path
        self.base64_image = self._encode_image_to_base64(image_path)
        # logger.info(f"Image loaded: {len(self.base64_image)} characters")
        # logger.info(f"Image path stored: {self.image_path}")

    def _setup_task_exception_handling(self) -> None:
        """Set up global task exception handling to prevent unhandled task warnings."""
        def task_exception_handler(task):
            """Handle exceptions from background tasks that might not be awaited."""
            try:
                # Try to get the exception
                exception = task.exception()
                if exception:
                    # Check if it's the common event loop closure issue
                    if isinstance(exception, RuntimeError) and "Event loop is closed" in str(exception):
                        logger.debug(f"Background task failed due to event loop closure - this is expected during shutdown")
                    else:
                        logger.warning(f"Background task failed with exception: {exception}")
            except Exception:
                # If we can't get the exception, just ignore
                pass

        # Set up exception handling for all current tasks
        try:
            loop = asyncio.get_event_loop()
            current_tasks = asyncio.all_tasks(loop)
            for task in current_tasks:
                if not task.done():
                    task.add_done_callback(task_exception_handler)
        except Exception as e:
            logger.debug(f"Could not set up task exception handling: {e}")

    async def execute(self, input_text: str, image_path: Optional[str] = None, log_save: Optional[LogSave] = None) -> str:
        """
        Execute the iterative refinement flow.

        Two-loop architecture:
        - Outer loop: Iterations between translator and reasoning (max 3)
        - Inner loop: Each agent's multi-step execution (max 3 steps per agent)

        Returns:
            Final answer from the reasoning agent
        """
        try:
            # Set up task exception handling to prevent unhandled task warnings
            self._setup_task_exception_handling()

            # Use external log_save if provided, otherwise use the instance one
            if log_save is not None:
                self.log_save = log_save

            # Parse and setup
            question, options, image_path = self._parse_input(input_text, image_path)
            self._setup_image(image_path)

            # logger.info(f"Starting iterative refinement flow (max {self.max_iterations} iterations)")
            logger.info(f"Question: {question}")
            # logger.info(f"Options: {options}")

            # Individual question logging is managed externally by the adapter
            session_id = None

            # Collect and display experiment configuration
            experiment_config = self._collect_experiment_config()
            # self._display_experiment_config(experiment_config)

            # Outer loop: Iterative refinement between agents #FIXME
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"=== Outer Loop Iteration {iteration}/{self.max_iterations} ===")

                # CRITICAL: Isolate each iteration to prevent bug propagation
                try:
                    # Reset iteration state to ensure clean slate
                    await self._prepare_iteration_isolation(iteration)

                    # Log iteration start if available
                    if hasattr(self.log_save, 'log_iteration_start'):
                        self.log_save.log_iteration_start(iteration)

                    # Reset memory for new iteration but preserve previous SIR context
                    self._reset_agent_memory_for_iteration(self.translator_agent, iteration)
                    self._reset_agent_memory_for_iteration(self.reasoning_agent, iteration)

                    # Set flow context and iteration info for agents
                    self.translator_agent.set_flow_iteration_context(f"{iteration}/{self.max_iterations}")
                    self.reasoning_agent.set_flow_iteration_context(f"{iteration}/{self.max_iterations}")
                    self.translator_agent.set_iteration_info(iteration, self.max_iterations)
                    self.reasoning_agent.set_iteration_info(iteration, self.max_iterations)

                    # Inner loop 1: Translator generates/refines SIR
                    sir_result = await self._run_translator_inner_loop(question, options, iteration)
                    if sir_result.startswith("Error:"):
                        return self._finalize_with_error(session_id, sir_result)

                    # Update SIR history before setting new current
                    self._validate_sir(sir_result, iteration)
                    self._update_sir_history(sir_result)

                    # Inner loop 2: Reasoning analyzes SIR and decides next action
                    reasoning_result, decision = await self._run_reasoning_inner_loop(question, options, iteration)
                    if reasoning_result.startswith("Error:"):
                        return self._finalize_with_error(session_id, reasoning_result)

                    # Note: Reasoning step logging is now handled by _log_agent_step with actual agent input
                    # The old log_reasoner_step created artificial input and is replaced by accurate logging

                    # Handle reasoning decision
                    if decision == "FINAL_ANSWER":
                        logger.info(f"Final answer obtained in {iteration} iteration(s)")
                        return self._finalize_with_success(session_id, reasoning_result)
                    elif decision == "CONTINUE_WITH_FEEDBACK" and iteration < self.max_iterations:
                        # Append feedback directly to current SIR for persistent context
                        self._append_feedback_to_sir(reasoning_result)

                        # Update SIR history before setting new current
                        self.previous_sir = self.current_sir   # ← 只做这一步
                        self._validate_sir(self.previous_sir, iteration)

                        logger.info(f"💬 Feedback received and appended to SIR, proceeding to iteration {iteration + 1}")
                        continue
                    else:
                        logger.warning(f"Max iterations reached or unclear response")
                        return self._finalize_with_success(session_id, reasoning_result)

                except Exception as iteration_error:
                    # Isolate iteration errors - don't let them affect subsequent iterations
                    clean_error = self._clean_error_message(str(iteration_error))
                    logger.error(f"💥 Iteration {iteration} failed with error: {clean_error}")

                    # If this is the last iteration or a critical error, fail completely
                    if iteration == self.max_iterations or self._is_critical_error(iteration_error):
                        logger.error(f"Critical error in iteration {iteration} or max iterations reached - failing completely")
                        return self._finalize_with_error(session_id, f"Iteration {iteration} failed: {clean_error}")

                    # For non-critical errors, try to recover and continue to next iteration
                    logger.warning(f"🔄 Attempting recovery - will try iteration {iteration + 1}")
                    await self._recover_from_iteration_failure(iteration, iteration_error)
                    continue

            # Max iterations reached
            final_result = "Error: Maximum iterations reached without final answer"
            return self._finalize_with_error(session_id, final_result)

        except Exception as e:
            # Clean error message to avoid printing base64 data
            error_msg = str(e)

            # Provide more specific error identification for common issues
            if "APIConnectionError" in error_msg:
                logger.error("API Connection Error in IterativeRefinementFlow - possible network/server issue")
                return "Flow execution failed: API Connection Error (network/server issue)"
            elif "RetryError" in error_msg:
                logger.error("Retry Error in IterativeRefinementFlow - API request failed after multiple attempts")
                return "Flow execution failed: API request failed after retries"
            else:
                clean_error = self._clean_error_message(error_msg)
                logger.error(f"Error in IterativeRefinementFlow: {clean_error}")
                return f"Flow execution failed: {clean_error}"

        finally:
            # CRITICAL: Ensure proper cleanup to prevent event loop closure issues
            try:
                await self._ensure_final_cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Final cleanup had minor issues: {cleanup_error}")
                # Don't propagate cleanup errors as they shouldn't affect the main result

    async def _ensure_final_cleanup(self) -> None:
        """Ensure all async resources are properly cleaned up before flow completion."""
        logger.debug("🧹 Performing final cleanup to prevent event loop issues")

        try:
            # Close HTTP clients properly
            await self._safe_close_http_clients()

            # Clean up agent resources
            for agent in [self.translator_agent, self.reasoning_agent]:
                if hasattr(agent, 'cleanup') and callable(agent.cleanup):
                    try:
                        await agent.cleanup()
                    except Exception as e:
                        logger.debug(f"Agent {agent.name} cleanup had minor issues: {e}")

            logger.debug("✅ Final cleanup completed successfully")

        except Exception as e:
            logger.debug(f"Final cleanup had issues: {e}")
            # Don't propagate cleanup errors

    async def _prepare_iteration_isolation(self, iteration: int) -> None:
        """
        Prepare clean state for iteration to prevent bug propagation.
        This ensures each iteration starts with a clean slate.
        """
        logger.info(f"🧹 Preparing iteration {iteration} isolation")

        try:
            # 1. Reset agent states to clean values
            await self._reset_agent_states()

            # 2. Clean up any corrupted connection pools
            await self._cleanup_connection_pools()

            # 3. Reset tool states
            await self._reset_tool_states()

            # 4. Clear any transient error flags
            self._clear_error_flags()

            logger.info(f"✅ Iteration {iteration} isolation prepared successfully")

        except Exception as e:
            logger.warning(f"⚠️ Iteration isolation preparation had minor issues: {e}")
            # Don't fail the iteration for isolation issues - just log and continue

    async def _reset_agent_states(self) -> None:
        """Reset both agents to clean, ready states."""
        # Reset agent states
        from app.schema import AgentState

        # Reset translator agent
        self.translator_agent.state = AgentState.IDLE
        self.translator_agent.current_step = 0
        self.translator_agent.tool_calls = []
        if hasattr(self.translator_agent, 'final_caption_json'):
            self.translator_agent.final_caption_json = None

        # Reset reasoning agent
        self.reasoning_agent.state = AgentState.IDLE
        self.reasoning_agent.current_step = 0
        self.reasoning_agent.tool_calls = []

        # Clear any retry flags
        if hasattr(self.translator_agent, 'in_tool_parsing_retry'):
            self.translator_agent.in_tool_parsing_retry = False
        if hasattr(self.reasoning_agent, 'in_tool_parsing_retry'):
            self.reasoning_agent.in_tool_parsing_retry = False

    async def _cleanup_connection_pools(self) -> None:
        """Clean up potentially corrupted HTTP connection pools with proper async cleanup."""
        try:
            # Properly close HTTP clients to prevent event loop closure issues
            await self._safe_close_http_clients()

            # Force LLM connection reset if available
            if hasattr(self.translator_agent, 'llm') and self.translator_agent.llm:
                if hasattr(self.translator_agent.llm, '_reset_connection_pool'):
                    await self.translator_agent.llm._reset_connection_pool()

            if hasattr(self.reasoning_agent, 'llm') and self.reasoning_agent.llm:
                if hasattr(self.reasoning_agent.llm, '_reset_connection_pool'):
                    await self.reasoning_agent.llm._reset_connection_pool()

        except Exception as e:
            logger.debug(f"Connection pool cleanup had minor issues: {e}")

    async def _safe_close_http_clients(self) -> None:
        """Safely close HTTP clients to prevent event loop closure race conditions."""
        import asyncio
        close_tasks = []

        try:
            # Find and close all HTTPX AsyncClient instances
            for agent in [self.translator_agent, self.reasoning_agent]:
                if hasattr(agent, 'llm') and agent.llm:
                    # Check for OpenAI client (which uses httpx internally)
                    if hasattr(agent.llm, '_client') and agent.llm._client:
                        client = agent.llm._client
                        # Check if it's an httpx AsyncClient or has aclose method
                        if hasattr(client, 'aclose'):
                            task = asyncio.create_task(self._safe_aclose(client, f"{agent.name}_llm_client"))
                            close_tasks.append(task)

            # Wait for all cleanup tasks with timeout to prevent hanging
            if close_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*close_tasks, return_exceptions=True),
                    timeout=5.0  # 5 second timeout for cleanup
                )

        except asyncio.TimeoutError:
            logger.warning("HTTP client cleanup timed out - some connections may not be properly closed")
        except Exception as e:
            logger.debug(f"HTTP client cleanup had issues: {e}")

    async def _safe_aclose(self, client, client_name: str) -> None:
        """Safely close an async client with proper error handling."""
        try:
            logger.debug(f"Closing {client_name}")
            await client.aclose()
            logger.debug(f"Successfully closed {client_name}")
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                logger.debug(f"Event loop already closed for {client_name} - ignoring")
            else:
                logger.warning(f"Runtime error closing {client_name}: {e}")
        except Exception as e:
            logger.warning(f"Error closing {client_name}: {e}")

    async def _reset_tool_states(self) -> None:
        """Reset tool states to prevent state contamination."""
        try:
            # Reset tool states for translator
            if hasattr(self.translator_agent, 'available_tools'):
                for tool_name, tool in self.translator_agent.available_tools.tool_map.items():
                    if hasattr(tool, 'reset_state'):
                        await tool.reset_state()

            # Reset tool states for reasoning agent
            if hasattr(self.reasoning_agent, 'available_tools'):
                for tool_name, tool in self.reasoning_agent.available_tools.tool_map.items():
                    if hasattr(tool, 'reset_state'):
                        await tool.reset_state()

        except Exception as e:
            logger.debug(f"Tool state reset had minor issues: {e}")

    def _clear_error_flags(self) -> None:
        """Clear transient error flags and state."""
        # Clear any error tracking flags
        if hasattr(self, '_iteration_error_count'):
            self._iteration_error_count = 0
        if hasattr(self, '_last_error_type'):
            self._last_error_type = None

    def _is_critical_error(self, error: Exception) -> bool:
        """
        Determine if an error is critical enough to stop all iterations.
        Non-critical errors allow recovery and continuation.
        """
        error_str = str(error).lower()

        # Critical errors that should stop all iterations
        critical_patterns = [
            'permission denied',
            'authentication failed',
            'authorization failed',
            'file not found',
            'module not found',
            'import error',
            'syntax error',
            'memory error',
            'disk full',
            'no space left'
        ]

        # Non-critical errors that allow recovery
        recoverable_patterns = [
            'connection error',
            'timeout',
            'api error',
            'retry error',
            'network error',
            'service unavailable',
            'rate limit',
            'token limit'
        ]

        # Check for critical errors first
        for pattern in critical_patterns:
            if pattern in error_str:
                return True

        # Check for recoverable errors
        for pattern in recoverable_patterns:
            if pattern in error_str:
                return False

        # Default: treat unknown errors as critical to be safe
        return True

    async def _recover_from_iteration_failure(self, failed_iteration: int, error: Exception) -> None:
        """
        Attempt recovery from iteration failure.
        This method implements strategies to recover from common error types.
        """
        error_str = str(error).lower()
        logger.info(f"🔧 Attempting recovery from iteration {failed_iteration} failure")

        try:
            # Strategy 1: Connection errors - wait and reset connections
            if any(pattern in error_str for pattern in ['connection error', 'network error', 'api error']):
                logger.info("🔗 Connection error detected - resetting connections")
                await self._cleanup_connection_pools()
                # Small delay to allow network recovery
                import asyncio
                await asyncio.sleep(1.0)

            # Strategy 2: Memory/token limit errors - clear memory aggressively
            elif any(pattern in error_str for pattern in ['token limit', 'memory error']):
                logger.info("💾 Memory/token error detected - aggressive memory cleanup")
                # Clear non-essential memory
                self.translator_agent.memory.messages = []
                self.reasoning_agent.memory.messages = []
                # Reset SIR history but keep the latest
                if hasattr(self, 'sir_history') and len(self.sir_history) > 1:
                    self.sir_history = [self.sir_history[-1]]  # Keep only the latest

            # Strategy 3: Tool/parsing errors - reset tool states
            elif any(pattern in error_str for pattern in ['tool', 'parse', 'json']):
                logger.info("🔧 Tool/parsing error detected - resetting tool states")
                await self._reset_tool_states()

            # Strategy 4: Generic recovery - full state reset
            else:
                logger.info("🧹 Generic error - performing full state cleanup")
                await self._reset_agent_states()
                await self._cleanup_connection_pools()
                await self._reset_tool_states()

            logger.info(f"✅ Recovery attempt for iteration {failed_iteration} completed")

        except Exception as recovery_error:
            logger.warning(f"⚠️ Recovery attempt had issues: {recovery_error}")
            # Don't fail on recovery issues - just continue

    async def _run_agent_with_retry(self, agent, agent_name: str, max_retries: int = 2):
        """
        Run agent with automatic retry for truncated tool calls.

        Args:
            agent: The agent instance to run
            agent_name: Name of the agent for logging
            max_retries: Maximum number of retry attempts

        Returns:
            Result from successful agent execution
        """
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"🔄 {agent_name} execution attempt {attempt + 1}/{max_retries + 1}")
                result = await agent.run()

                # Check if tool calls were parsed successfully with valid arguments
                if self._has_invalid_tool_calls(result):
                    if attempt < max_retries:
                        logger.warning(f"⚠️ Detected invalid/truncated tool calls in {agent_name}, retrying...")

                        # Clean recent failed attempt from memory to prevent pollution
                        self._clean_failed_attempt_from_memory(agent)

                        # Provide concise, specific feedback
                        feedback = (
                            f"🔧 RETRY {attempt + 1}: Previous tool call was incomplete. "
                            f"Please provide a complete tool call with valid JSON arguments."
                        )
                        agent.update_memory("user", feedback)

                        # Set retry flag to prevent conflicting step continuation messages
                        if hasattr(agent, 'set_tool_parsing_retry_mode'):
                            agent.set_tool_parsing_retry_mode(True)

                        continue
                    else:
                        logger.warning(f"⚠️ {agent_name} still showing invalid tool calls after {max_retries} retries")

                # Reset retry flag after successful execution
                if hasattr(agent, 'set_tool_parsing_retry_mode'):
                    agent.set_tool_parsing_retry_mode(False)

                return result

            except Exception as e:
                error_str = str(e).lower()

                # Check for tool argument errors (likely due to truncation)
                if any(pattern in error_str for pattern in [
                    "missing.*required.*argument",
                    "missing 1 required positional argument",
                    "unexpected keyword argument",
                    "arguments.*missing",
                    "invalid arguments"
                ]) and attempt < max_retries:

                    logger.warning(f"⚠️ Tool argument error in {agent_name} (attempt {attempt + 1}): {e}")

                    # Clean recent failed attempt from memory to prevent pollution
                    self._clean_failed_attempt_from_memory(agent)

                    # Provide concise, specific feedback about the tool error
                    feedback = (
                        f"🔧 RETRY {attempt + 1}: Tool error - {str(e)[:100]}... "
                        f"Please provide complete tool call with all required arguments."
                    )
                    agent.update_memory("user", feedback)

                    # Set retry flag to prevent conflicting step continuation messages
                    if hasattr(agent, 'set_tool_parsing_retry_mode'):
                        agent.set_tool_parsing_retry_mode(True)

                    continue

                # Reset retry flag and re-raise non-recoverable errors
                if hasattr(agent, 'set_tool_parsing_retry_mode'):
                    agent.set_tool_parsing_retry_mode(False)
                raise e

        # If we get here, all retries failed - ensure retry flag is reset
        if hasattr(agent, 'set_tool_parsing_retry_mode'):
            agent.set_tool_parsing_retry_mode(False)

        logger.error(f"❌ {agent_name} failed after {max_retries + 1} attempts")
        raise Exception(f"{agent_name} execution failed after {max_retries + 1} attempts")

    def _setup_agent_memory(self, agent, question: str, options: Optional[List[str]], iteration: int) -> str:
        import re, json
        agent_type = agent.name.lower()

        if agent_type == "translator":
            options_text = f"\n\nOptions: {', '.join(options)}" if options else ""

            if iteration == 1:
                # iteration 1 不走这里，由 _run_translator_direct_vision 处理
                translator_input = (
                    f"You are a visual captioner. Describe what you see visually.\n"
                    f"Question: {question}{options_text}\n\nImage file: {self.image_path}"
                )
                agent.update_memory("user", translator_input, base64_image=self.base64_image)
                logger.info("Added initial request to translator memory")
                return translator_input

            else:
                # 改为只打log确认一下：
                if hasattr(agent, 'current_sir') and agent.current_sir:
                    logger.info(f"✅ translator.current_sir retained from previous iteration ({len(agent.current_sir)} chars)")

                # 解析 Still need 和建议工具，动态设置 first_step_prompt
                still_need = ""
                suggested_tool = "smart_grid_caption"  # 默认工具

                need_match = re.search(
                    r'Still need:\s*(.+?)(?:\n|$)',
                    self.current_sir or "",
                    re.IGNORECASE
                )
                if need_match:
                    still_need = need_match.group(1).strip()
                    # 只取第一个工具名，防止模型列了多个
                    still_need_first = still_need.split(',')[0].split(' and ')[0].strip()
                    still_need_lower = still_need_first.lower()
                    if "ocr" in still_need_lower:
                        suggested_tool = "ocr"
                    elif "read_table" in still_need_lower:
                        suggested_tool = "read_table"
                    elif "smart_grid_caption" in still_need_lower:
                        suggested_tool = "smart_grid_caption"

                logger.info(f"🎯 Refinement target: still_need='{still_need}', suggested_tool='{suggested_tool}'")

                # 动态覆盖 first_step_prompt，强制 step1 直接调指定工具
                agent.first_step_prompt = (
                    f"This is refinement iteration {iteration}. The reasoning agent needs:\n\n"
                    f'"{still_need}"\n\n'
                    f"Your ONLY task:\n"
                    f"1. Call `{suggested_tool}` RIGHT NOW to get the requested information\n"
                    f"2. Do NOT call any other tool before `{suggested_tool}`\n"
                    f"3. After the tool result comes back, update the SIR and call `terminate_and_output_caption`\n\n"
                    f"Execute `{suggested_tool}` immediately."
                )

                merged_translator_input = (
                    f"You are refining your visual description based on reasoning feedback.\n"
                    f"Question: {question}{options_text}\n"
                    f"Image file: {self.image_path}\n\n"
                    f"Current SIR with reasoning feedback (iteration {iteration - 1}):\n"
                    f"{self.current_sir}\n\n"
                    f"IMPROVEMENT TASK:\n"
                    f"1. Read the 'REASONING FEEDBACK' section — focus on 'Still need'\n"
                    f"2. Use `{suggested_tool}` to get the specific information requested\n"
                    f"3. Append the new detail to the existing SIR — do NOT rewrite from scratch\n"
                    f"4. Call `terminate_and_output_caption` with the updated SIR"
                )
                agent.update_memory("user", merged_translator_input, base64_image=self.base64_image)
                logger.info(
                    f"Added SIR improvement task to translator memory "
                    f"(iteration {iteration}, tool={suggested_tool}, SIR: {len(self.current_sir)} chars)"
                )
                return merged_translator_input

        elif agent_type == "text_only_reasoning":
            options_text = f"\n\nOptions: {', '.join(options)}" if options else ""

            if not (hasattr(self, 'current_sir') and self.current_sir):
                context_input = f"Question: {question}"
                if options:
                    context_input += f"\n\nOptions: {', '.join(options)}"
                agent.update_memory("user", context_input)
                logger.warning(f"No SIR available for reasoning agent in iteration {iteration}")
                return context_input

            # 只取 global_caption，剥离 translator 内部元数据 # 避免无关字段干扰 reasoning，同时节省 token
            sir_text = self.current_sir
            try:
                sir_json = json.loads(self.current_sir)
                if "global_caption" in sir_json:
                    sir_text = sir_json["global_caption"]
                    logger.info("✂️ Extracted global_caption from SIR JSON for reasoning")
            except (json.JSONDecodeError, TypeError):
                # 非 JSON 格式（如含 REASONING FEEDBACK 的纯文本），直接用原文
                pass

            merged_reasoning_input = (
                f"Question: {question}{options_text}\n\n"
                f"Visual analysis from translator (iteration {iteration}):\n"
                f"{sir_text}\n\n"
                f"REASONING TASK:\n"
                f"1. Analyze the visual description above\n"
                f"2. Use python_execute for any calculations needed\n"
                f"3. If confident, call terminate_and_answer with your final answer\n"
                f"4. If you need more visual details, call terminate_and_ask_translator with:\n"
                f"   - Preliminary answer: your current best answer\n"
                f"   - Confidence: high/medium/low\n"
                f"   - Still need: ONE tool only (OCR/smart_grid_caption/read_table): ONE specific task"
            )
            agent.update_memory("user", merged_reasoning_input)
            logger.info(
                f"Added reasoning task to agent memory "
                f"(iteration {iteration}, SIR: {len(sir_text)} chars)"
            )
            return merged_reasoning_input

        return f"Question: {question}"

    def _log_system_prompt(self, agent_name: str, system_prompt: str, iteration: int) -> None:
        """Log agent's system prompt"""
        if hasattr(self.log_save, 'add_individual_question_message') and self.log_save.current_question:
            self.log_save.add_individual_question_message(
                role="system",
                content=system_prompt,
                metadata={
                    "agent": agent_name.lower(),
                    "iteration": iteration,
                    "prompt_type": "system_prompt"
                }
            )

    def _log_user_prompt(self, agent_name: str, user_input: str, iteration: int, prompt_type: str = "user_input") -> None:
        """Log user prompt/question to agent"""
        if hasattr(self.log_save, 'add_individual_question_message') and self.log_save.current_question:
            self.log_save.add_individual_question_message(
                role="user",
                content=user_input,
                metadata={
                    "agent": agent_name.lower(),
                    "iteration": iteration,
                    "prompt_type": prompt_type
                }
            )

    def _log_agent_response(self, agent_name: str, agent_response: str, iteration: int, tool_calls: Optional[List] = None) -> None:
        """Log agent's thinking + action response"""
        if hasattr(self.log_save, 'add_individual_question_message') and self.log_save.current_question:
            self.log_save.add_individual_question_message(
                role=agent_name.lower(),
                content=agent_response,
                tool_calls=tool_calls,
                metadata={
                    "iteration": iteration,
                    "agent_type": agent_name.lower(),
                    "response_length": len(agent_response),
                    "has_tool_calls": bool(tool_calls)
                }
            )

    def _log_tool_execution(self, tool_name: str, tool_input: str, tool_output: str, iteration: int) -> None:
        """Log tool execution and results"""
        if hasattr(self.log_save, 'add_individual_question_message') and self.log_save.current_question:
            self.log_save.add_individual_question_message(
                role="tool",
                content=f"Tool: {tool_name}\nInput: {tool_input}\nOutput: {tool_output}",
                metadata={
                    "tool_name": tool_name,
                    "iteration": iteration,
                    "input_length": len(tool_input),
                    "output_length": len(tool_output)
                }
            )

    def _clean_failed_attempt_from_memory(self, agent) -> None:
        """Clean recent failed assistant responses from agent memory to prevent pollution during retries"""
        try:
            messages = agent.memory.messages
            if len(messages) < 2:
                return

            # Remove the last assistant message if it contained failed tool calls
            # This prevents accumulation of failed attempts in memory
            if messages[-1].role == "assistant":
                logger.debug(f"Removing failed assistant message from {agent.name} memory")
                messages.pop()

            # Also remove any trailing tool messages from the failed attempt
            while messages and messages[-1].role == "tool":
                logger.debug(f"Removing failed tool message from {agent.name} memory")
                messages.pop()

        except Exception as e:
            logger.warning(f"Failed to clean memory for {agent.name}: {e}")
            # Don't let memory cleaning failures break the retry mechanism

    def _has_invalid_tool_calls(self, result: str) -> bool:
        """
        Check if the result contains invalid or incomplete tool calls.
        """
        from app.agent.toolcall import parse_tool_calls_multiple_formats

        # 如果已经是成功的终止信号，直接返回 False，不做检查
        if any(signal in result for signal in [
            "FINAL ANSWER:",
            "has been completed successfully",
            "The reasoning process has been completed",
            "FEEDBACK from reasoning agent:",
            "terminate_and_answer",
            "terminate_and_ask_translator",
            "terminate_and_output_caption",
        ]):
            return False

        # 只有当出现明确的工具调用标记时才进一步检查
        explicit_tool_patterns = ['<tool_call>', '```json\n{', '"name":', '"arguments":']
        if not any(pattern in result for pattern in explicit_tool_patterns):
            return False

        try:
            tool_calls = parse_tool_calls_multiple_formats(result)

            if not tool_calls:
                if any(indicator in result for indicator in ['<tool_call>', '"name":']):
                    logger.warning("Tool call patterns found but parsing failed - likely truncated")
                    return True
                return False

            for tool_call in tool_calls:
                if not hasattr(tool_call, 'function') or not tool_call.function:
                    return True
                args = tool_call.function.arguments
                if not args or args.strip() in ['{}', '""', "''", '']:
                    logger.warning(f"Tool call '{tool_call.function.name}' has empty arguments")
                    return True

            return False

        except Exception as e:
            logger.warning(f"Error parsing tool calls: {e}")
            return True

    async def _run_translator_inner_loop(self, question: str, options: List[str], iteration: int) -> str:
        try:
            translator = self.translator_agent
            logger.info(f"🌐 [Flow Iteration {iteration}/{self.max_iterations}] Starting TRANSLATOR inner loop")
            translator._flow_context = self
            translator.set_flow_iteration_context(f"{iteration}/{self.max_iterations}")
            translator.set_iteration_info(iteration, self.max_iterations)
            translator.current_step = 0

            # FIXME: Iteration 1 专用 - 直接单次 VLM call，绕过 agent loop 和工具调用，避免初始 SIR 过短导致的工具调用解析失败问题
            if iteration == 1:
                result = await self._run_translator_direct_vision(question, options)
                logger.info(f"📝 Flow SIR set from direct VLM call ({len(result)} chars)")
                return result

            actual_translator_input = self._setup_agent_memory(translator, question, options, iteration)

            # 保存此时的干净值（上一轮 global_caption）
            sir_snapshot = getattr(translator, 'current_sir', None)

            try:
                logger.info(f"Starting translator inner loop (iteration {iteration})")
                raw_result = await self._run_agent_with_retry(translator, "Translator")
                result = self._extract_translator_result(translator, raw_result)
                logger.info(f"📝 Translator result extracted ({len(result)} chars)")
                return result
            except Exception as e:
                # 恢复到入口时的干净值，而不是 None
                translator.current_sir = sir_snapshot
                logger.warning(f"Restored translator.current_sir to snapshot ({len(sir_snapshot) if sir_snapshot else 0} chars)")
                raise

        except Exception as e:
            error_msg = str(e)
            if "APIConnectionError" in error_msg:
                logger.error("API Connection Error in translator inner loop")
                return "Error: Translator inner loop failed: API Connection Error (network/server issue)"
            elif "RetryError" in error_msg:
                logger.error("Retry Error in translator inner loop")
                return "Error: Translator inner loop failed: API request failed after retries"
            elif "TokenLimitExceeded" in error_msg:
                logger.error("Token Limit Error in translator inner loop")
                return "Error: Translator inner loop failed: Token limit exceeded"
            else:
                clean_error = self._clean_error_message(error_msg)
                logger.error(f"Error in translator inner loop: {clean_error}")
                return f"Error: Translator inner loop failed: {clean_error}"


    async def _run_translator_direct_vision(self, question: str, options: List[str]) -> str:
        """Iteration 1 专用：不走 agent loop，直接单次 VLM call，不调用任何工具"""
        translator = self.translator_agent
        options_text = f"\n\nOptions: {', '.join(options)}" if options else ""
        prompt = (
            f"Describe all visually observable details relevant to this question.\n"
            f"Be objective, do not infer the answer.\n\n"
            f"Question: {question}{options_text}"
        )
        logger.info("🔍 [Iteration 1] Direct VLM call (no tools, no inner loop)")
        response = await translator.llm.ask(
            messages=[{
                "role": "user",
                "content": prompt,
                "base64_image": self.base64_image,
            }],
            system_msgs=[{
                "role": "system",
                "content": DIRECT_VISION_PROMPT,
            }],
            stream=False,
        )
        logger.info(f"✅ Direct VLM response ({len(response)} chars)")
        return response

    def _extract_translator_result(self, translator, raw_result: str) -> str:
        import json

        # 优先用工具输出的JSON
        if hasattr(translator, 'final_caption_json') and translator.final_caption_json:
            result_str = translator.final_caption_json
        else:
            caption_data = self._extract_caption_json(raw_result)
            if not caption_data:
                logger.info("Using raw step output")
                translator.current_sir = raw_result  # fallback
                return raw_result
            result_str = json.dumps(caption_data, ensure_ascii=False, indent=2)

        # 提取干净的 global_caption 写入 translator.current_sir
        try:
            data = json.loads(result_str)
            clean_caption = data.get("global_caption", "")
            if clean_caption:
                translator.current_sir = clean_caption
                logger.info(f"✅ translator.current_sir = clean global_caption ({len(clean_caption)} chars)")
        except Exception:
            pass

        # ❌ 删掉原来的 supplementation（--- Assistant Analysis ---）步骤
        logger.info("Extracted JSON from raw_result")
        return result_str


    def _extract_caption_json(self, text: str) -> Optional[dict]:
        """用平衡括号匹配提取最后一个含 global_caption 的 JSON 对象
        注意：在字符串内部的括号不参与深度计数
        """
        import json

        positions = [i for i, c in enumerate(text) if c == '{']

        for start in reversed(positions):
            depth = 0
            in_string = False
            escape_next = False

            for i, c in enumerate(text[start:], start):
                if escape_next:
                    escape_next = False
                    continue
                if c == '\\' and in_string:
                    escape_next = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue  # 字符串内部的 { } 不计入深度

                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        if '"global_caption"' in candidate:
                            try:
                                return json.loads(candidate)
                            except json.JSONDecodeError:
                                break
                        break

        return None

    async def _run_reasoning_inner_loop(self, question: str, options: List[str], iteration: int) -> Tuple[str, str]:
        """Run reasoning agent's inner loop and determine decision.

        Returns:
            Tuple of (result, decision) where decision is one of:
            - FINAL_ANSWER: Agent provided final answer
            - CONTINUE_WITH_FEEDBACK: Agent wants more info from translator
            - MAX_ITERATIONS_OR_UNCLEAR: Unclear response or max iterations
        """
        try:
            reasoning = self.reasoning_agent

            logger.info(f"🧠 [Flow Iteration {iteration}/{self.max_iterations}] Starting REASONING inner loop")

            # Set flow context for structured logging
            reasoning._flow_context = self
            reasoning.set_flow_iteration_context(f"{iteration}/{self.max_iterations}")
            reasoning.set_iteration_info(iteration, self.max_iterations)

            # Reset step counter for new iteration
            reasoning.current_step = 0

            # # Iteration 1：直接 LLM call，不走 agent loop，不调用工具
            # if iteration == 1:
            #     result, decision = await self._run_reasoning_direct(question, options)
            #     return result, decision

            # Setup reasoning agent memory using modular helper and capture actual input
            actual_reasoning_input = self._setup_agent_memory(reasoning, question, options, iteration)

            logger.info(f"✅ Added SIR to reasoning agent memory (iteration {iteration}, length: {len(self.current_sir)} chars)")

            # Log preview of SIR content for debugging
            if hasattr(self, 'current_sir') and self.current_sir:
                sir_preview = self.current_sir[:300] + "..." if len(self.current_sir) > 300 else self.current_sir
                logger.info(f"📄 SIR Preview for Reasoning: {sir_preview}")

            # logger.info("Starting reasoning inner loop")

            # Run reasoning inner loop with retry logic for truncated tool calls
            # Agent will see: initial question + all previous SIRs + previous reasoning attempts
            result = await self._run_agent_with_retry(reasoning, "Reasoning")

            # Determine decision based on result
            decision = self._determine_reasoning_decision(result, iteration)

            # Log reasoning execution with improved structure following ReAct pattern
            if iteration == 1 and hasattr(reasoning, 'system_prompt') and reasoning.system_prompt:
                self._log_system_prompt("Reasoning", reasoning.system_prompt, iteration)
            # self._log_user_prompt("Reasoning", actual_reasoning_input, iteration)
            self._log_agent_response("Reasoning", result, iteration)

            return result, decision

        except Exception as e:
            # Clean error message to avoid printing base64 data
            error_msg = str(e)

            # Provide more specific error identification for common issues
            if "APIConnectionError" in error_msg:
                logger.error("API Connection Error in reasoning inner loop - possible network/server issue")
                return "Error: Reasoning inner loop failed: API Connection Error (network/server issue)", "MAX_ITERATIONS_OR_UNCLEAR"
            elif "RetryError" in error_msg:
                logger.error("Retry Error in reasoning inner loop - API request failed after multiple attempts")
                return "Error: Reasoning inner loop failed: API request failed after retries", "MAX_ITERATIONS_OR_UNCLEAR"
            elif "TokenLimitExceeded" in error_msg:
                logger.error("Token Limit Error in reasoning inner loop")
                return "Error: Reasoning inner loop failed: Token limit exceeded", "MAX_ITERATIONS_OR_UNCLEAR"
            else:
                # Generic error but clean the message
                clean_error = self._clean_error_message(error_msg)
                logger.error(f"Error in reasoning inner loop: {clean_error}")
                return f"Error: Reasoning inner loop failed: {clean_error}", "MAX_ITERATIONS_OR_UNCLEAR"

    async def _run_reasoning_direct(self, question: str, options: List[str]) -> Tuple[str, str]:
        """Iteration 1 专用：不走 agent loop，直接单次 LLM call，不调用任何工具"""
        import re
        reasoning = self.reasoning_agent
        options_text = f"\n\nOptions: {', '.join(options)}" if options else ""
        prompt = (
            f"Question: {question}{options_text}\n\n"
            f"Visual description (SIR):\n{self.current_sir}"
        )
        logger.info("🧠 [Iteration 1] Direct reasoning call (no tools, no inner loop)")
        response = await reasoning.llm.ask(
            messages=[{"role": "user", "content": prompt}],
            system_msgs=[{"role": "system", "content": DIRECT_REASONING_PROMPT}],
            stream=False,
        )
        logger.info(f"✅ Direct reasoning response ({len(response)} chars)")

        # 提取 Preliminary answer（只取第一行）
        answer_match = re.search(r'Preliminary answer:\s*(.+)', response, re.IGNORECASE)
        preliminary_answer = answer_match.group(1).strip() if answer_match else response.strip()

        # 提取 Still need（作为 feedback）
        need_match = re.search(r'Still need:\s*(.+)', response, re.IGNORECASE)
        still_need = need_match.group(1).strip() if need_match else ""

        # 兜底：preliminary_answer 过长说明模型给了图像描述而非数学值，强制降级
        if len(preliminary_answer) > 50:
            logger.warning(f"⚠️ Preliminary answer too long ({len(preliminary_answer)} chars), forcing CONTINUE_WITH_FEEDBACK")
            feedback = (
                f"Preliminary answer: unclear\n"
                f"Still need: smart_grid_caption: analyze the full image to extract all numerical values, labels and measurements needed to solve the problem"
            )
            return feedback, "CONTINUE_WITH_FEEDBACK"

        response_lower = response.lower()
        if "confidence: high" in response_lower:
            formatted = (
                f"FINAL ANSWER: {preliminary_answer}\n\n"
                f"Confidence: high\n\n"
                f"Reasoning: Based on initial visual analysis."
            )
            logger.info(f"🎯 Iteration 1 reasoning: high confidence → FINAL_ANSWER: {preliminary_answer}")
            return formatted, "FINAL_ANSWER"
        else:
            feedback = (
                f"Preliminary answer: {preliminary_answer}\n"
                f"Still need: {still_need}"
            )
            logger.info(f"🔄 Iteration 1 reasoning: medium/low confidence → CONTINUE_WITH_FEEDBACK")
            return feedback, "CONTINUE_WITH_FEEDBACK"

    def _determine_reasoning_decision(self, result: str, iteration: int) -> str:
        """Determine the decision from reasoning agent result."""
        if self._is_final_answer(result):
            # iter1 且还有后续迭代时，强制继续以获取更精准的视觉信息
            if iteration == 1 and self.max_iterations > 1:
                logger.info(
                    "🔄 iter1 called terminate_and_answer but forcing CONTINUE_WITH_FEEDBACK "
                    "to allow translator refinement"
                )
                # 把 final answer 的内容作为 feedback 传给 translator
                return "CONTINUE_WITH_FEEDBACK"
            return "FINAL_ANSWER"
        elif self._is_feedback(result):
            return "CONTINUE_WITH_FEEDBACK"
        else:
            return "MAX_ITERATIONS_OR_UNCLEAR"


    def _validate_sir(self, sir_result: str, iteration: int) -> None:
        """Validate SIR format and log accordingly."""
        try:
            import json
            if sir_result.startswith('{') and sir_result.endswith('}'):
                parsed_json = json.loads(sir_result)
                if 'global_caption' in parsed_json:
                    logger.info(f"Generated valid JSON SIR (iteration {iteration})")
                else:
                    logger.warning(f"SIR missing 'global_caption' field (iteration {iteration})")
            else:
                logger.info(f"Generated text-based SIR (iteration {iteration})")
        except json.JSONDecodeError:
            logger.warning(f"SIR is not valid JSON format (iteration {iteration})")

    def _is_final_answer(self, response: str) -> bool:
        """Check if the response contains a final answer."""
        response_lower = response.lower()
        return (
            response.strip().upper().startswith("FINAL ANSWER:") or
            "terminate_and_answer" in response_lower or
            "has been completed successfully" in response_lower
        )

    def _is_feedback(self, response: str) -> bool:
        """Check if the response contains feedback for refinement."""
        response_lower = response.lower()
        return (
            response.strip().upper().startswith("FEEDBACK:") or
            "terminate_and_ask_translator" in response_lower or
            "missing" in response_lower or
            "need" in response_lower or
            "unclear" in response_lower or
            "more information" in response_lower
        )

    def _finalize_with_success(self, session_id: Optional[str], result: str) -> str:
        """Finalize logging session with successful result."""
        return result

    def _finalize_with_error(self, session_id: Optional[str], error_result: str) -> str:
        """Finalize logging session with error result."""
        return error_result

    def _collect_experiment_config(self) -> Dict[str, Any]:
        """Collect experiment configuration from agents and flow settings."""
        try:
            # Get model configurations from agents
            translator_config = {}
            reasoning_config = {}

            if hasattr(self.translator_agent, 'llm'):
                llm = self.translator_agent.llm
                translator_config = {
                    "model": getattr(llm, 'model', 'Unknown'),
                    "api_type": getattr(llm, 'api_type', 'Unknown'),
                    "temperature": getattr(llm, 'temperature', 'Unknown'),
                    "max_tokens": getattr(llm, 'max_tokens', 'Unknown'),
                    "base_url": getattr(llm, 'base_url', 'N/A'),
                }
                # Add vLLM specific parameters if available
                if hasattr(llm, 'provider'):
                    provider = llm.provider
                    if hasattr(provider, 'tensor_parallel_size'):
                        translator_config["tensor_parallel_size"] = provider.tensor_parallel_size
                    if hasattr(provider, 'gpu_memory_utilization'):
                        translator_config["gpu_memory_utilization"] = provider.gpu_memory_utilization
                    if hasattr(provider, 'model_name'):
                        translator_config["provider_model"] = provider.model_name

            if hasattr(self.reasoning_agent, 'llm'):
                llm = self.reasoning_agent.llm
                reasoning_config = {
                    "model": getattr(llm, 'model', 'Unknown'),
                    "api_type": getattr(llm, 'api_type', 'Unknown'),
                    "temperature": getattr(llm, 'temperature', 'Unknown'),
                    "max_tokens": getattr(llm, 'max_tokens', 'Unknown'),
                    "base_url": getattr(llm, 'base_url', 'N/A'),
                }
                # Add vLLM specific parameters if available
                if hasattr(llm, 'provider'):
                    provider = llm.provider
                    if hasattr(provider, 'tensor_parallel_size'):
                        reasoning_config["tensor_parallel_size"] = provider.tensor_parallel_size
                    if hasattr(provider, 'gpu_memory_utilization'):
                        reasoning_config["gpu_memory_utilization"] = provider.gpu_memory_utilization
                    if hasattr(provider, 'model_name'):
                        reasoning_config["provider_model"] = provider.model_name

            # Flow configuration
            flow_config = {
                "max_iterations": self.max_iterations,
                "flow_type": "iterative_refinement"
            }

            # System information
            import torch
            system_config = {
                "gpu_available": torch.cuda.is_available() if torch is not None else False,
                "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
            }

            return {
                "models": {
                    "translator": translator_config,
                    "reasoning": reasoning_config
                },
                "flow": flow_config,
                "system": system_config
            }

        except Exception as e:
            logger.warning(f"Failed to collect experiment config: {e}")
            return {
                "models": {"translator": {}, "reasoning": {}},
                "flow": {"max_iterations": self.max_iterations, "flow_type": "iterative_refinement"},
                "system": {"gpu_available": "Unknown", "gpu_memory": "Unknown"}
            }

    def _clean_error_message(self, error_msg: str) -> str:
        """Clean error message to remove base64 image data and other sensitive content."""
        # Remove base64 image data
        clean_msg = error_msg.replace('"base64_image":', '"base64_image": "[EXCLUDED]",')

        # Also handle cases where base64 strings might appear
        import re
        # Remove long base64-like strings (30+ characters of base64 pattern)
        clean_msg = re.sub(r'"[A-Za-z0-9+/]{30,}={0,2}"', '"[BASE64_DATA_EXCLUDED]"', clean_msg)

        # Truncate extremely long error messages
        if len(clean_msg) > 1000:
            clean_msg = clean_msg[:1000] + "... [TRUNCATED]"

        return clean_msg

    def _display_experiment_config(self, experiment_config: Dict[str, Any]):
        """Display experiment configuration in terminal."""
        logger.info("🔧 EXPERIMENT CONFIGURATION")
        logger.info("=" * 50)

        # Models configuration
        models = experiment_config.get("models", {})
        if models:
            logger.info("📋 MODELS:")
            for agent_name, model_config in models.items():
                if model_config:
                    logger.info(f"  • {agent_name.upper()}:")
                    logger.info(f"    - Model: {model_config.get('model', 'N/A')}")
                    logger.info(f"    - API Type: {model_config.get('api_type', 'N/A')}")
                    logger.info(f"    - Temperature: {model_config.get('temperature', 'N/A')}")
                    logger.info(f"    - Max Tokens: {model_config.get('max_tokens', 'N/A')}")
                    logger.info(f"    - Base URL: {model_config.get('base_url', 'N/A')}")

                    # vLLM specific parameters
                    if model_config.get('api_type') == 'vllm':
                        logger.info(f"    - Provider Model: {model_config.get('provider_model', 'N/A')}")
                        logger.info(f"    - GPU Memory: {model_config.get('gpu_memory_utilization', 'N/A')}")
                        logger.info(f"    - Tensor Parallel: {model_config.get('tensor_parallel_size', 'N/A')}")

        # Flow configuration
        flow_config = experiment_config.get("flow", {})
        if flow_config:
            logger.info("")
            logger.info("⚙️ FLOW SETTINGS:")
            logger.info(f"  • Max Iterations: {flow_config.get('max_iterations', 'N/A')}")
            logger.info(f"  • Flow Type: {flow_config.get('flow_type', 'N/A')}")

        # System information
        system_config = experiment_config.get("system", {})
        if system_config:
            logger.info("")
            logger.info("💻 SYSTEM INFO:")
            logger.info(f"  • GPU Available: {system_config.get('gpu_available', 'N/A')}")
            logger.info(f"  • GPU Memory: {system_config.get('gpu_memory', 'N/A')}")

        logger.info("=" * 50)