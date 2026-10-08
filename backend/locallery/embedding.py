"""Official Transformers preprocessing, projected pooling, and normalization."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .config import PREPROCESS, Config


def normalize(values, dimensions=768):
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (dimensions,) or not np.isfinite(vector).all():
        raise ValueError(f"Embedding must contain {dimensions} finite values")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding must have a finite, nonzero norm")
    return vector / norm


class Embedder:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.fingerprint = ""

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor
        from transformers.utils.hub import cached_file, extract_commit_hash

        config = self.config
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = config.dtype
        if dtype == "auto":
            dtype = (
                "bfloat16"
                if device.startswith("cuda") and torch.cuda.is_bf16_supported()
                else "float32"
            )
        resolved_dtype = getattr(torch, dtype)
        cache_dir = str(config.cache_dir) if config.cache_dir is not None else None
        config_file = cached_file(
            config.model, "config.json", revision=config.revision, cache_dir=cache_dir
        )
        commit = extract_commit_hash(config_file, None)
        revision = commit or config.revision
        model_config = AutoConfig.from_pretrained(
            config.model, revision=revision, cache_dir=cache_dir
        )
        if getattr(model_config, "model_type", None) != "embedding_gemma2":
            raise ValueError("embedding.model must be an EmbeddingGemma 2 checkpoint")
        # Only vision and text are used; avoid loading the independent audio tower.
        model_config.audio_config = None
        processor = AutoProcessor.from_pretrained(
            config.model, revision=revision, cache_dir=cache_dir
        )
        model = (
            AutoModel.from_pretrained(
                config.model,
                revision=revision,
                cache_dir=cache_dir,
                config=model_config,
                dtype=resolved_dtype,
            )
            .to(device)
            .eval()
        )
        local_identity = []
        if Path(config.model).is_dir():
            for file in sorted(Path(config.model).rglob("*")):
                if file.is_file():
                    stat = file.stat()
                    local_identity.append(
                        (
                            str(file.relative_to(config.model)),
                            stat.st_size,
                            stat.st_mtime_ns,
                        )
                    )
        identity = [
            "transformers-pooling-v1",
            config.model,
            commit or config.revision,
            local_identity,
            device,
            dtype,
            version("transformers"),
            PREPROCESS,
            768,
        ]
        self.fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        self.processor, self.model, self.device = processor, model, device
        print(
            f"EmbeddingGemma 2 loaded on {device} ({dtype}) · revision {commit or config.revision or 'local'}",
            flush=True,
        )

    def embed(
        self,
        query: str | None = None,
        image: str | None = None,
        video: str | None = None,
    ):
        self.load()
        import torch

        content = []
        if video:
            content.append({"type": "video"})
        if image:
            content.append({"type": "image", "url": str(image)})
        if query:
            content.append(
                {"type": "text", "text": f"task: search result | query: {query}"}
            )
        if not content:
            raise ValueError("An image, video, or query is required")
        messages = [{"role": "user", "content": content}]
        if video:
            from .videos import load_video_cache

            frames, metadata = load_video_cache(video)
            prompt = self.processor.apply_chat_template(messages, tokenize=False)
            inputs = self.processor(
                text=prompt,
                videos=[frames],
                video_metadata=[metadata],
                do_sample_frames=False,
                add_timestamps=self.config.video.add_timestamps,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(self.device)
        if inputs["input_ids"].shape[-1] > 8192:
            raise ValueError("Query exceeds the model's 8,192-token context")
        with torch.inference_mode():
            tokens = self.model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return normalize(pooled[0].cpu().numpy())
