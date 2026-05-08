"""Tokenizer wrapper. Pure-CPU; cost is negligible."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Tokenizer:
    def __init__(self, artifact_dir: Path, max_length: int = 256):
        from tokenizers import Tokenizer as HFTokenizer

        path = artifact_dir / "tokenizer.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Tokenizer must be packed alongside the QNN binaries."
            )
        self._tok = HFTokenizer.from_file(str(path))
        self._tok.enable_padding(length=max_length)
        self._tok.enable_truncation(max_length=max_length)
        self.max_length = max_length

    def encode(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (input_ids, attention_mask) shaped (1, max_length), int32."""
        enc = self._tok.encode(prompt)
        ids = np.asarray(enc.ids, dtype=np.int32)[None, :]
        mask = np.asarray(enc.attention_mask, dtype=np.int32)[None, :]
        return ids, mask
