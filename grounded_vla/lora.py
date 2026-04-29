"""LoRA fine-tuning entry point (Stretch Goal H3).

This module is a thin wrapper around HuggingFace PEFT that's set up for the
synthetic instruction-image corpus. It is deliberately minimal we don't
try to reimplement a trainer; we rely on `transformers.Trainer` so the
risk surface stays small. If GPU resources don't materialize, this file
stays unused the rest of the project still runs.

Usage::

    from grounded_vla.lora import train_lora
    train_lora(
        jsonl_path="data/synthetic/synthetic.jsonl",
        images_dir="data/synthetic/images",
        output_dir="checkpoints/llava-lora-run-1",
    )
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class LoRAConfig:
    base_model: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    learning_rate: float = 2e-4
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    max_seq_len: int = 1024


def train_lora(
    jsonl_path: Path | str,
    images_dir: Path | str,
    output_dir: Path | str,
    config: Optional[LoRAConfig] = None,
) -> Path:
    """Fine-tune LLaVA-1.6 with LoRA on the synthetic dataset.

    Returns the directory containing the trained adapter.
    """
    try:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            LlavaNextForConditionalGeneration,
            Trainer,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise RuntimeError(
            "LoRA fine-tuning needs the [gpu] extra: `pip install -e .[gpu]`"
        ) from e

    config = config or LoRAConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(config.base_model)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        config.base_model, quantization_config=bnb, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = _load_synthetic_dataset(jsonl_path, images_dir, processor, config.max_seq_len)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        report_to=[],
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(output_dir)
    _log.info("LoRA adapter saved to %s", output_dir)
    return output_dir


def _load_synthetic_dataset(jsonl_path, images_dir, processor, max_seq_len):
    """Build a torch Dataset from the synthetic JSONL.

    Each example is rendered as an instruction + target-action string so
    the model learns to emit actions in our expected format.
    """
    from PIL import Image
    import torch
    from torch.utils.data import Dataset as TorchDataset

    jsonl_path = Path(jsonl_path)
    images_dir = Path(images_dir)
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]

    class _DS(TorchDataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = rows[i]
            img_path = Path(r["image_path"])
            if not img_path.is_absolute():
                img_path = images_dir.parent / img_path
            image = Image.open(img_path).convert("RGB")
            gold = r["gold_actions"][0]
            target = (
                "Thought: Looking at the image, I will perform the grounded action.\n"
                "Action: " + json.dumps(gold)
            )
            prompt = r["instruction"]
            chat = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
                {"role": "assistant", "content": target},
            ]
            prompt_str = processor.apply_chat_template(chat, add_generation_prompt=False)
            enc = processor(
                images=image,
                text=prompt_str,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_seq_len,
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = item["input_ids"].clone()
            return item

    return _DS()
