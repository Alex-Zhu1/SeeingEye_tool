import json
from app.tool.base import BaseTool


_TERMINATE_AND_OUTPUT_CAPTION_DESCRIPTION = """Terminate visual analysis and output the final visual description in JSON format.

Use this tool when:
- You have completed your visual analysis of the image
- You have gathered all necessary visual information through direct observation or tools
- You are ready to output a comprehensive, objective visual description

Example call:
terminate_and_output_caption(
    global_caption="A triangle ABC with three semicircles drawn outward on each side. The semicircle on AB is the largest. Two black shaded crescent-shaped regions appear on the semicircles of AC and BC. Labels A, B, C mark the vertices. No numerical values are visible.",
    confidence="high",
    summary_of_this_turn="1. **Initial Visual Analysis**: Focused on the shaded crescent regions near vertices A and B as requested. 2. **Tool Usage**: Used smart_grid_caption to examine spatial relationships between semicircles and shaded regions. 3. **Feedback Integration**: Reasoning agent requested confirmation of shaded region boundaries — confirmed they are bounded by arcs of the leg semicircles and the hypotenuse semicircle. 4. **Final Refinement**: Appended shaded region detail to existing SIR."
)

This signals that the visual analysis is complete."""


class TerminateAndOutputCaption(BaseTool):
    name: str = "terminate_and_output_caption"
    description: str = _TERMINATE_AND_OUTPUT_CAPTION_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "global_caption": {
                "type": "string",
                "description": (
                    "A comprehensive, objective description of ALL visible elements: "
                    "text, shapes, colors, labels, numbers, spatial relationships, layout, arrows, tables. "
                    "Rules:\n"
                    "(1) Describe only what is VISIBLE — no inference, no calculation, no domain knowledge.\n"
                    "(2) Preserve all text verbatim including blanks, '?', typos, casing.\n"
                    "(3) Do NOT answer the question or draw conclusions.\n"
                    "(4) Do NOT include phrases like 'suggesting', 'therefore', 'based on configuration'.\n"
                    "(5) If this is a refinement iteration, append the new detail to the previous description "
                    "— do NOT rewrite from scratch."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "mid", "high"],
                "description": (
                    "'high' = all visible elements captured comprehensively.\n"
                    "'mid' = good analysis but some regions unclear.\n"
                    "'low' = incomplete, image unclear or key elements missing."
                ),
            },
            "summary_of_this_turn": {
                "type": "string",
                "description": (
                    "Structured summary following EXACTLY this format:\n"
                    "1. **Initial Visual Analysis**: What you directly observed from the image "
                    "in this iteration — if feedback was provided, focus on the specific region "
                    "or element requested, not a general re-description of the whole image.\n"
                    "2. **Tool Usage**: Which tool you used and what it returned (or 'No tool used').\n"
                    "3. **Feedback Integration**: What specific detail was requested by the reasoning agent "
                    "and how you addressed it (or 'No feedback received in this iteration').\n"
                    "4. **Final Refinement**: What was appended or updated compared to the previous SIR "
                    "(or 'Built from scratch' if this is the first iteration).\n\n"
                    "RULES:\n"
                    "- Each section must start with its number and bold title exactly as shown.\n"
                    "- Section 1 must NOT repeat the entire previous SIR — only describe new observations.\n"
                    "- Do NOT merge sections.\n"
                    "- Do NOT include calculations, conclusions, or inferences in any section."
                ),
            }
        },
        "required": ["global_caption", "confidence", "summary_of_this_turn"],
    }

    async def execute(self, global_caption: str, confidence: str, summary_of_this_turn: str, **kwargs) -> str:
        """Terminate visual analysis and output final JSON caption"""
        caption_data = {
            "global_caption": global_caption,
            "confidence": confidence,
            "summary_of_this_turn": summary_of_this_turn
        }
        return json.dumps(caption_data, indent=2, ensure_ascii=False)