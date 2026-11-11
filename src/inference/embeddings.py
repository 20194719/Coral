"""Encode full texts in a shared normalized embedding space."""

import numpy as np


class Encoder:
    def __init__(self, model_path, device, batch_size, dimension=2560):
        """Initialize model state."""
        self.path, self.device, self.batch_size = str(model_path), device, batch_size
        self.dimension = dimension
        self.model = None
        self.memo = {}

    def _load(self):
        """Load the embedding model."""
        if self.model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer

        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.model = SentenceTransformer(
            self.path,
            device=self.device,
            local_files_only=True,
            model_kwargs={"torch_dtype": dtype, "attn_implementation": "sdpa"},
            tokenizer_kwargs={"padding_side": "left"},
        )
        limits = [
            self.model[0].auto_model.config.max_position_embeddings,
            self.model.tokenizer.model_max_length,
        ]
        self.model.max_seq_length = min(int(v) for v in limits if 0 < v < 10**9)
        if self.model.get_sentence_embedding_dimension() != self.dimension:
            raise ValueError(f"Expected {self.dimension}-dimensional embeddings")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def __call__(self, texts):
        """Evaluate the requested inputs."""
        texts = list(texts)
        missing = list(dict.fromkeys(t for t in texts if t not in self.memo))
        if missing:
            self._load()
            print(f"[embedding] encoding {len(missing)} texts", flush=True)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            tokens = self.model.tokenizer(batch, padding=False, truncation=False)[
                "input_ids"
            ]
            if any(len(ids) > self.model.max_seq_length for ids in tokens):
                raise ValueError(
                    "Embedding input exceeds model capacity; full text was not truncated"
                )
            values = self.model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                prompt="",
            )
            values = np.asarray(values, dtype=np.float32)
            if (
                values.shape != (len(batch), self.dimension)
                or not np.isfinite(values).all()
            ):
                raise ValueError("Invalid embedding output")
            self.memo.update(zip(batch, values))

            if start and start % (100 * self.batch_size) == 0:
                print(f"[embedding] {start}/{len(missing)}", flush=True)
        return (
            np.stack([self.memo[t] for t in texts])
            if texts
            else np.empty((0, self.dimension), np.float32)
        )


def encode(cfg, texts):
    """Encode document or predicate texts."""
    return cfg.encoder(texts)
