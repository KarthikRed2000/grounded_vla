"""LLaVA-1.6 (7B) backend.

Loads ``llava-hf/llava-v1.6-mistral-7b-hf`` lazily so that environments
without torch/transformers installed can still import this module. The
heavy dependencies are imported inside the class only when weights are
actually requested.

Usage::

    from grounded_vla.backends.llava import LLaVABackend
    be = LLaVABackend(device="cuda", quantize="4bit")
    be.warmup()
    out = be.generate(prompt, image=pil_img)
"""
from __future__ import annotations

from typing import Optional

from PIL import Image

from .base import Backend, BackendError, GenerationConfig


class LLaVABackend(Backend):
    name = "llava-1.6-7b"
    supports_vision = True

    DEFAULT_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        quantize: Optional[str] = "4bit",
        hf_cache_dir: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.quantize = quantize
        self.hf_cache_dir = hf_cache_dir
        self._model = None
        self._processor = None

    def warmup(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoProcessor,
                LlavaNextForConditionalGeneration,
            )
        except ImportError as e:
            raise BackendError(
                "LLaVA backend requires transformers + torch. "
                "Install with `pip install -e .[gpu]`."
            ) from e

        load_kwargs: dict = {"cache_dir": self.hf_cache_dir}
        if self.quantize == "4bit":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise BackendError("bitsandbytes required for 4-bit LLaVA") from e
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        elif self.quantize == "8bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = self.device

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, cache_dir=self.hf_cache_dir
        )
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, **load_kwargs
        )
        self._model.eval()

    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        if image is None:
            raise BackendError("LLaVABackend requires an image; use a text-only backend instead")
        self._ensure_loaded()
        config = config or GenerationConfig()

        import torch

        # LLaVA-1.6 chat template expects an <image> token + user text.
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_str = self._processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = self._processor(images=image, text=prompt_str, return_tensors="pt").to(
            self._model.device
        )
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.temperature > 0,
                temperature=max(config.temperature, 1e-5),
                top_p=config.top_p,
            )
        text = self._processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        return text

    def close(self) -> None:
        if self._model is not None:
            try:
                import torch

                del self._model
                del self._processor
                self._model = None
                self._processor = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
