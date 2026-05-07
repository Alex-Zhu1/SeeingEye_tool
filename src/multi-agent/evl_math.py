#!/usr/bin/env python3
"""
Euclid30K Benchmark Evaluation
Dataset: https://huggingface.co/datasets/LiamLian0727/Euclid30K

Columns:
    problem : str   - question text with <image> placeholders
    images  : list  - PIL images (1 or more per problem)
    answer  : str   - ground truth answer
"""

import os
import re
import json
import argparse
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from datasets import load_dataset
from PIL import Image

# ── your existing flow executor ──────────────────────────────────────────────
# 在其他 import 之后加
os.environ["OPENAI_API_KEY"] = "dummy"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from app.flow.flow_executor import FlowExecutor


# ============================================================================
# 1. Answer normalisation & matching
# ============================================================================

def normalise(text: str) -> str:
    """Strip LaTeX, whitespace and punctuation for loose comparison."""
    if not text:
        return ""
    s = text.strip()
    # remove LaTeX delimiters
    s = re.sub(r'\$', '', s)
    s = re.sub(r'\\boxed\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()


def is_correct(predicted: str, ground_truth: str) -> bool:
    """
    Three-tier matching:
      1. Exact match after normalisation
      2. Ground truth is contained in predicted (handles verbose answers)
      3. Numeric match with 2% tolerance
    """
    pred = normalise(predicted)
    gt   = normalise(ground_truth)

    if not gt:
        return False

    # tier 1: exact
    if pred == gt:
        return True

    # tier 2: substring
    if gt in pred:
        return True

    # tier 3: numeric
    try:
        pred_num = float(re.sub(r'[^\d.\-eE]', '', pred))
        gt_num   = float(re.sub(r'[^\d.\-eE]', '', gt))
        if gt_num != 0:
            return abs(pred_num - gt_num) / abs(gt_num) < 0.02
        else:
            return abs(pred_num - gt_num) < 1e-6
    except (ValueError, TypeError):
        pass

    return False


# ============================================================================
# 2. Image helpers
# ============================================================================

def save_images(images: list, tmp_dir: Path) -> list:
    """Save PIL images to disk and return their paths."""
    paths = []
    for i, img in enumerate(images):
        if isinstance(img, dict) and "bytes" in img:
            from io import BytesIO
            img = Image.open(BytesIO(img["bytes"]))
        path = str(tmp_dir / f"img_{i}.png")
        img.save(path)
        paths.append(path)
    return paths


def build_flow_input(problem: str, image_paths: list) -> str:
    """
    Replace <image> placeholders with [image:path] hints,
    then append image_path: for the flow parser.
    """
    text = problem.strip()

    for path in image_paths:
        text = text.replace("<image>", f"[image:{path}]", 1)

    primary = image_paths[0] if image_paths else ""
    if primary:
        text += f"\nimage_path:{primary}"

    return text


# ============================================================================
# 3. Main evaluator
# ============================================================================

async def evaluate(
    split: str = "validation",
    max_samples: Optional[int] = None,
    max_iterations: int = 3,
    output_path: str = "euclid_results.json",
    start_idx: int = 0,
):
    print(f"📥 Loading Euclid30K ({split} split)...")
    dataset = load_dataset("LiamLian0727/Euclid30K", split=split)

    end_idx = min(start_idx + max_samples, len(dataset)) if max_samples else len(dataset)
    dataset = dataset.select(range(start_idx, end_idx))

    total = len(dataset)
    print(f"📊 Evaluating {total} samples  |  max_iterations={max_iterations}")

    executor = FlowExecutor()

    results = []
    correct = 0
    errors  = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for idx, sample in enumerate(dataset):
            global_idx = start_idx + idx
            problem    = sample["problem"]
            images     = sample["images"]
            gt_answer  = sample["answer"]

            print(f"\n[{idx+1}/{total}] idx={global_idx}")
            print(f"  Q: {problem[:120].replace(chr(10),' ')}...")
            print(f"  A: {gt_answer}")

            img_paths  = save_images(images, tmp_dir)
            flow_input = build_flow_input(problem, img_paths)

            try:
                result    = await executor.execute_async(flow_input)
                response  = result["response"]
                success   = result["success"]
                exec_time = result["execution_time_seconds"]
            except Exception as e:
                response  = f"EXCEPTION: {e}"
                success   = False
                exec_time = 0.0
                errors   += 1

            correct_flag = is_correct(response, gt_answer) if success else False
            if correct_flag:
                correct += 1

            running_acc = correct / (idx + 1) * 100
            status = "✅" if correct_flag else "❌"
            print(f"  {status} Pred: {response[:120]}")
            print(f"  Running acc: {running_acc:.1f}%  ({correct}/{idx+1})")

            results.append({
                "idx":         global_idx,
                "problem":     problem,
                "gt_answer":   gt_answer,
                "response":    response,
                "correct":     correct_flag,
                "success":     success,
                "exec_time_s": exec_time,
            })

            # checkpoint every 50 samples
            if (idx + 1) % 50 == 0:
                _save(output_path, results, correct, idx + 1, errors)
                print(f"  💾 Checkpoint saved ({idx+1}/{total})")

    _save(output_path, results, correct, total, errors)

    final_acc = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"✅ Final accuracy : {final_acc:.2f}%  ({correct}/{total})")
    print(f"❌ Errors         : {errors}")
    print(f"💾 Saved to       : {output_path}")
    return final_acc


def _save(path: str, results: list, correct: int, total: int, errors: int):
    out = {
        "meta": {
            "timestamp":    datetime.now().isoformat(),
            "total":        total,
            "correct":      correct,
            "accuracy_pct": round(correct / total * 100, 2) if total else 0,
            "errors":       errors,
        },
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ============================================================================
# 4. CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SeeingEye on Euclid30K")
    p.add_argument("--split",          default="validation",
                   choices=["train", "validation"],
                   help="Dataset split (default: validation)")
    p.add_argument("--max_samples",    type=int, default=None,
                   help="Limit number of samples (default: all)")
    p.add_argument("--start_idx",      type=int, default=0,
                   help="Start index for resuming (default: 0)")
    p.add_argument("--max_iterations", type=int, default=3,
                   help="Flow max outer iterations (default: 3)")
    p.add_argument("--output",         default="euclid_results.json",
                   help="Output JSON file path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(evaluate(
        split          = args.split,
        max_samples    = args.max_samples,
        max_iterations = args.max_iterations,
        output_path    = args.output,
        start_idx      = args.start_idx,
    ))