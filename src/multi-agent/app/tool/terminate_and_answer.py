from app.tool.base import BaseTool

_TERMINATE_AND_ANSWER_DESCRIPTION = """Terminate the reasoning process and provide a final answer.

Use this tool when you have sufficient information to confidently answer the question.

IMPORTANT: All your detailed thinking and derivation should be done BEFORE calling this tool 
(e.g., via python_execute or prior reasoning steps). 
The `reasoning` field here is only a brief summary — do NOT re-derive or repeat your full analysis."""


class TerminateAndAnswer(BaseTool):
    name: str = "terminate_and_answer"
    description: str = _TERMINATE_AND_ANSWER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Your final answer. For multiple choice, include only the option letter. For open-ended, a concise but complete response.",
            },
            "confidence": {
                "type": "string",
                "description": "Your confidence level in this answer.",
                "enum": ["high", "medium", "low"],
            },
            "reasoning": {
                "type": "string",
                "description": "One or two sentences summarizing the key evidence from the SIR that led to this answer. Do NOT repeat your full derivation or re-analyze — that should happen before calling this tool.",
                "maxLength": 300,
            },
        },
        "required": ["answer", "confidence", "reasoning"],
    }

    async def execute(self, answer: str, confidence: str, reasoning: str) -> str:
        reasoning = (reasoning or "").strip()[:300]  # 兜底
        return (
            f"FINAL ANSWER: {answer}\n\n"
            f"Confidence: {confidence}\n\n"
            f"Reasoning: {reasoning}\n\n"
            f"The reasoning process has been completed successfully."
        )