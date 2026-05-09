from app.tool.base import BaseTool


_TERMINATE_AND_ASK_TRANSLATOR_DESCRIPTION = """Terminate current reasoning and request ONE specific visual detail from the translator.

Use when:
- Any numerical value in the answer has NOT been verified via python_execute
- The visual description uses vague language ("appears to be", "seems like", "possibly")
- You cannot definitively rule out all other options
- Confidence is below 0.95

Required fields:
- preliminary_answer: concrete value (e.g. "6", "16°", "B") — NOT an image description
- confidence: high / medium / low (see strict definitions below)
- still_need: ONE tool + ONE task, format "<tool>: <task>"
  Tools: OCR / smart_grid_caption / read_table
  ✅ "OCR: extract the numerical value of BC"
  ✅ "smart_grid_caption: read exact Y-axis values"
  ❌ "OCR and smart_grid_caption: ..." (multiple tools forbidden)
  ❌ "more details" (too vague)"""


class TerminateAndAskTranslator(BaseTool):
    name: str = "terminate_and_ask_translator"
    description: str = _TERMINATE_AND_ASK_TRANSLATOR_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "preliminary_answer": {
                "type": "string",
                "description": (
                    "Your current best concrete answer, e.g. '6', '16°', 'B'. "
                    "Must be a specific value — NOT a description of the image. "
                    "If you cannot form any concrete answer yet, write 'unknown'."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": (
                    "STRICT definitions — do NOT upgrade your confidence:\n"
                    "'high': You have a concrete answer AND have verified it via python_execute "
                    "AND the visual description provides specific unambiguous evidence. "
                    "Still need = minor verification only (e.g. confirm one label).\n"
                    "'medium': You have a reasonable guess but the visual description is incomplete, "
                    "vague, or uses language like 'appears to be' / 'seems like'. "
                    "Calculation has NOT been verified.\n"
                    "'low': Key visual information is clearly missing. "
                    "You cannot form a concrete answer without more visual detail.\n\n"
                    "DEFAULT to 'medium' if in doubt — do NOT use 'high' unless all conditions are met."
                ),
            },
            "still_need": {
                "type": "string",
                "description": (
                    "ONE tool + ONE specific task. Format: '<tool>: <specific task>'\n"
                    "Tools: OCR / smart_grid_caption / read_table\n"
                    "✅ 'OCR: extract the exact numerical value labeled on BC'\n"
                    "✅ 'smart_grid_caption: read exact Y-axis tick values in the bar chart'\n"
                    "✅ 'read_table: extract all numerical values from the data table'\n"
                    "❌ multiple tools in one request\n"
                    "❌ vague requests like 'more details' or 'confirm the answer'"
                ),
            }
        },
        "required": ["preliminary_answer", "confidence", "still_need"],
    }

    async def execute(self, preliminary_answer: str, confidence: str, still_need: str) -> str:
        return (
            f"Preliminary answer: {preliminary_answer}\n"
            f"Confidence: {confidence}\n"
            f"Still need: {still_need}"
        )