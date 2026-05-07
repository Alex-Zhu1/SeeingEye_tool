"""OCR tool for extracting text from images using Qwen-VL-OCR (DashScope)."""

import os
import base64
import mimetypes
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from app.tool.base import BaseTool, ToolResult


class OCR(BaseTool):
    """OCR tool using Qwen-VL-OCR via DashScope's OpenAI-compatible endpoint."""

    name: str = "ocr"
    description: str = (
        "Extract text content from image files using Qwen-VL-OCR. "
        "Supports multilingual text recognition. An optional `prompt` "
        "can be provided to guide what to extract (e.g. only the table, "
        "only the title, return as JSON, etc.)."
    )

    # Tool parameter schema (visible to the LLM agent)
    parameters: dict = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the image file (supports relative and absolute paths)",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Optional custom instruction for OCR extraction, e.g. "
                    "'extract only the table content as markdown' or "
                    "'return key fields as JSON'. If omitted, all visible "
                    "text in the image is extracted as plain text."
                ),
            },
        },
        "required": ["image_path"],
    }

    # ---- Qwen-VL-OCR configuration ----
    model: str = "qwen-vl-ocr-latest"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # Default prompt used by qwen-vl-ocr when no custom prompt is provided.
    default_prompt: str = (
        "Please output only the text content from the image "
        "without any additional descriptions or formatting."
    )
    # Pixel thresholds recommended in the Qwen-VL-OCR docs.
    min_pixels: int = 32 * 32 * 3
    max_pixels: int = 32 * 32 * 8192
    # Network timeout (seconds)
    request_timeout: float = 60.0

    async def execute(
        self,
        image_path: str,
        prompt: Optional[str] = None,
    ) -> ToolResult:
        """Run Qwen-VL-OCR on a local image file."""
        try:
            # 1. Validate file
            image_file = Path(image_path)
            if not image_file.exists():
                return ToolResult(error=f"Image file does not exist: {image_path}")

            allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}
            if image_file.suffix.lower() not in allowed_extensions:
                return ToolResult(
                    error=(
                        f"Unsupported image format: {image_file.suffix}. "
                        f"Supported formats: {', '.join(sorted(allowed_extensions))}"
                    )
                )

            # 2. Validate API key
            # api_key = os.getenv("DASHSCOPE_API_KEY")
            api_key = "sk-4381e45ba46441b98e0c958fa66e32b7"
            if not api_key:
                return ToolResult(
                    error="DASHSCOPE_API_KEY environment variable is not set."
                )

            # 3. Encode image as base64 data URL
            mime_type, _ = mimetypes.guess_type(str(image_file))
            if mime_type is None:
                ext = image_file.suffix.lower().lstrip(".")
                mime_type = f"image/{'jpeg' if ext == 'jpg' else ext}"

            with open(image_file, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            data_url = f"data:{mime_type};base64,{b64_data}"

            # 4. Call Qwen-VL-OCR via OpenAI-compatible API
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.request_timeout,
            )

            completion = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                                # Per Qwen-VL-OCR docs, these sit alongside
                                # `image_url`, not inside it.
                                "min_pixels": self.min_pixels,
                                "max_pixels": self.max_pixels,
                            },
                            {
                                "type": "text",
                                "text": prompt if prompt else self.default_prompt,
                            },
                        ],
                    }
                ],
            )

            extracted_text = (completion.choices[0].message.content or "").strip()

            output_text = f"OCR Results:\nExtracted Text:\n{extracted_text}\n"
            return ToolResult(output=output_text)

        except FileNotFoundError:
            return ToolResult(error=f"File not found: {image_path}")
        except Exception as e:
            return ToolResult(error=f"OCR processing error: {type(e).__name__}: {e}")