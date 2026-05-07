import argparse
import asyncio
from pathlib import Path

from app.flow.flow_executor import FlowExecutor
from app.logger import logger


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=False)
    parser.add_argument("--image",  type=str, required=False)
    args = parser.parse_args()

    prompt = args.prompt if args.prompt else input("Enter prompt: ")
    if not prompt.strip():
        return

    image_path = (args.image or "").strip()
    if image_path and not Path(image_path).exists():
        logger.warning(f"Image not found: {image_path}")
        return

    # config 自动读取：
    #   TranslatorAgent      ← [llm.translator_api]
    #   TextOnlyReasoningAgent ← [llm.reasoning_api]
    executor = FlowExecutor()

    # _parse_input() 期望的格式
    flow_input = prompt
    if image_path:
        flow_input += f"\nimage_path:{image_path}"

    logger.info("Running: Translator → Reasoner...")
    result = await executor.execute_async(flow_input)
    logger.info(f"FINAL ANSWER:\n{result['response']}")
    logger.info(f"Time: {result['execution_time_seconds']}s")


if __name__ == "__main__":
    asyncio.run(main())