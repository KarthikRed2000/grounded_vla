"""One-time preprocessing: pull Mind2Web from HuggingFace and write a JSONL
plus a folder of screenshots. After running once, subsequent eval runs use
the local files (no network, no HF auth).

Usage::

    python scripts/prepare_mind2web.py \
        --split test_task \
        --out-dir data/mind2web \
        --limit 200
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_task")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install `datasets` and `huggingface_hub` first: pip install -e .[data]")

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{args.split}.jsonl"

    ds = load_dataset("osunlp/Mind2Web", split=args.split, streaming=True)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(tqdm(ds, desc="mind2web")):
            if args.limit is not None and i >= args.limit:
                break
            task_id = str(ex.get("annotation_id") or ex.get("task_id") or i)
            # Some rows ship screenshots as base64 bytes; decode and save.
            shot = ex.get("screenshot")
            if isinstance(shot, (bytes, bytearray)):
                Image.open(io.BytesIO(shot)).save(images_dir / f"{task_id}.png")
            elif isinstance(shot, str) and shot.startswith(("iVBOR", "/9j/")):
                Image.open(io.BytesIO(base64.b64decode(shot))).save(images_dir / f"{task_id}.png")

            row = {
                "task_id": task_id,
                "instruction": ex.get("confirmed_task") or ex.get("task"),
                "image_path": f"images/{task_id}.png",
                "text": ex.get("cleaned_html"),
                "gold_actions": _coerce_actions(ex.get("actions") or ex.get("action_reprs") or []),
                "max_steps": 20,
                "meta": {"domain": ex.get("domain"), "website": ex.get("website")},
            }
            f.write(json.dumps(row) + "\n")

    print(f"wrote {jsonl_path}")


def _coerce_actions(actions) -> list[dict]:
    out = []
    for a in actions:
        if isinstance(a, str) and "->" in a:
            left, right = a.rsplit("->", 1)
            out.append({"type": right.strip().lower(), "target": left.strip()})
        elif isinstance(a, dict):
            op = (a.get("operation") or {}).get("op", "click")
            out.append({"type": op.lower(), "target": a.get("raw_html"), "value": (a.get("operation") or {}).get("value")})
    return out


if __name__ == "__main__":
    main()
