"""Z-Image-Turbo orchestrator.

Loads pre-compiled QNN context binaries on demand, runs the 8-step flow-
matching loop, and frees each ORT session before loading the next stage so
that the 16 GB phone RAM stays under budget.

All heavy ops run on the Hexagon HTP via QNNExecutionProvider. CPU is only
used for the tokenizer, the scheduler arithmetic, and PNG encoding.
"""

from __future__ import annotations

import gc
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .npu import HtpInfo, NpuUnavailable, assert_htp_alive, ensure_env
from .ort_session import make_session
from .scheduler import FlowMatchScheduler, SchedulerConfig
from .tokenizer import Tokenizer


log = logging.getLogger("zimage")

DEFAULT_CACHE = Path.home() / ".cache" / "zimage" / "qnn"

# Required artifacts in the QNN cache directory.
ARTIFACT_TEXT_ENCODER = "text_encoder.qnn.bin"
ARTIFACT_TRANSFORMER_IN = "transformer_in.qnn.bin"
ARTIFACT_TRANSFORMER_OUT = "transformer_out.qnn.bin"
ARTIFACT_VAE_DECODER = "vae_decoder.qnn.bin"
ARTIFACT_TOKENIZER = "tokenizer.json"
ARTIFACT_PIPELINE_CFG = "pipeline_config.json"

BLOCK_PATTERN = re.compile(r"^transformer_block_(\d{2,3})\.qnn\.bin$")


@dataclass(frozen=True)
class PipelineConfig:
    latent_channels: int       # VAE latent channels (typically 16 for Z-Image)
    vae_scale_factor: int      # Spatial downsampling factor of the VAE (typically 8)
    patch_size: int            # DiT patch size (typically 2)
    hidden_size: int           # DiT hidden size
    text_max_length: int       # Tokenizer truncation length
    eviction_window: int       # Max simultaneous block sessions kept resident

    @classmethod
    def from_dir(cls, path: Path) -> "PipelineConfig":
        cfg_path = path / ARTIFACT_PIPELINE_CFG
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"Missing {cfg_path}. The compile step must emit pipeline_config.json."
            )
        cfg = json.loads(cfg_path.read_text())
        return cls(
            latent_channels=int(cfg["latent_channels"]),
            vae_scale_factor=int(cfg["vae_scale_factor"]),
            patch_size=int(cfg["patch_size"]),
            hidden_size=int(cfg["hidden_size"]),
            text_max_length=int(cfg.get("text_max_length", 256)),
            eviction_window=int(cfg.get("eviction_window", 4)),
        )


def discover_artifacts(cache_dir: Path) -> dict[str, Path]:
    """Return {artifact_name: path} for every required artifact, or raise.

    Listed deterministically; transformer block files are returned in numeric
    order under the key "transformer_blocks".
    """
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Artifact cache {cache_dir} does not exist. "
            "Compile the QNN context binaries on a workstation and copy them here. "
            "See compile/README.md."
        )

    required = [
        ARTIFACT_TEXT_ENCODER,
        ARTIFACT_TRANSFORMER_IN,
        ARTIFACT_TRANSFORMER_OUT,
        ARTIFACT_VAE_DECODER,
        ARTIFACT_TOKENIZER,
        ARTIFACT_PIPELINE_CFG,
    ]
    found: dict[str, Path] = {}
    missing: list[str] = []
    for name in required:
        p = cache_dir / name
        if p.is_file():
            found[name] = p
        else:
            missing.append(name)

    blocks: list[Path] = []
    for entry in sorted(cache_dir.iterdir()):
        m = BLOCK_PATTERN.match(entry.name)
        if m and entry.is_file():
            blocks.append(entry)
    if not blocks:
        missing.append("transformer_block_NN.qnn.bin (none found)")

    if missing:
        raise FileNotFoundError(
            "Missing artifacts in {}:\n  - {}".format(
                cache_dir, "\n  - ".join(missing)
            )
        )
    found["transformer_blocks"] = blocks  # type: ignore[assignment]
    return found


def _free(*sessions) -> None:
    for s in sessions:
        del s
    gc.collect()


def _patchify(latent: np.ndarray, patch: int) -> np.ndarray:
    """Reshape (1, C, H, W) → (1, (H/p)*(W/p), C*p*p)."""
    b, c, h, w = latent.shape
    assert h % patch == 0 and w % patch == 0
    x = latent.reshape(b, c, h // patch, patch, w // patch, patch)
    x = x.transpose(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // patch) * (w // patch), c * patch * patch)


def _unpatchify(seq: np.ndarray, patch: int, h: int, w: int, c: int) -> np.ndarray:
    """Inverse of _patchify."""
    b = seq.shape[0]
    x = seq.reshape(b, h // patch, w // patch, c, patch, patch)
    x = x.transpose(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, h, w)


class ZImagePipeline:
    """Single-shot pipeline. Re-instantiate per generation; sessions are freed."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE):
        ensure_env()
        self.htp: HtpInfo = assert_htp_alive()
        self.cache_dir = cache_dir
        self.artifacts = discover_artifacts(cache_dir)
        self.cfg = PipelineConfig.from_dir(cache_dir)
        self.scheduler_cfg = SchedulerConfig.from_dir(cache_dir)
        self.tokenizer = Tokenizer(cache_dir, max_length=self.cfg.text_max_length)

    # ---- Stages ---------------------------------------------------------

    def _encode_text(self, prompt: str) -> np.ndarray:
        ids, mask = self.tokenizer.encode(prompt)
        sess = make_session(self.artifacts[ARTIFACT_TEXT_ENCODER], self.htp)
        try:
            outputs = sess.run(None, {"input_ids": ids, "attention_mask": mask})
        finally:
            _free(sess)
        return outputs[0].astype(np.float16)

    def _run_transformer(
        self,
        latent_seq: np.ndarray,
        text_embeds: np.ndarray,
        timestep: np.ndarray,
    ) -> np.ndarray:
        """One forward pass of the DiT: in → blocks (lazy) → out."""
        in_sess = make_session(self.artifacts[ARTIFACT_TRANSFORMER_IN], self.htp)
        try:
            x = in_sess.run(
                None,
                {
                    "latent": latent_seq,
                    "text_embeds": text_embeds,
                    "timestep": timestep,
                },
            )[0]
        finally:
            _free(in_sess)

        block_paths: list[Path] = self.artifacts["transformer_blocks"]  # type: ignore[assignment]
        window = max(1, self.cfg.eviction_window)
        # Lazy-load blocks in groups of `window` to bound resident memory.
        for start in range(0, len(block_paths), window):
            group = block_paths[start : start + window]
            sessions = [make_session(p, self.htp) for p in group]
            try:
                for sess in sessions:
                    x = sess.run(None, {"hidden": x, "text_embeds": text_embeds})[0]
            finally:
                _free(*sessions)

        out_sess = make_session(self.artifacts[ARTIFACT_TRANSFORMER_OUT], self.htp)
        try:
            v = out_sess.run(None, {"hidden": x})[0]
        finally:
            _free(out_sess)
        return v

    def _decode_vae(self, latent: np.ndarray) -> np.ndarray:
        sess = make_session(self.artifacts[ARTIFACT_VAE_DECODER], self.htp)
        try:
            image = sess.run(None, {"latent": latent})[0]
        finally:
            _free(sess)
        return image  # (1, 3, H, W) in [-1, 1]

    # ---- Public API -----------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        steps: int = 8,
        size: int = 1024,
        seed: int | None = None,
    ) -> np.ndarray:
        """Run end-to-end. Returns an HxWx3 uint8 RGB array."""
        if size % (self.cfg.vae_scale_factor * self.cfg.patch_size) != 0:
            raise ValueError(
                f"--size {size} is not a multiple of "
                f"vae_scale_factor*patch_size = "
                f"{self.cfg.vae_scale_factor * self.cfg.patch_size}"
            )

        rng = np.random.default_rng(seed)
        latent_h = size // self.cfg.vae_scale_factor
        latent_w = size // self.cfg.vae_scale_factor
        seq_len = (latent_h // self.cfg.patch_size) * (latent_w // self.cfg.patch_size)

        scheduler = FlowMatchScheduler(self.scheduler_cfg, steps, seq_len)

        log.info("Encoding text")
        text_embeds = self._encode_text(prompt)

        log.info("Initializing latent (size=%d, seq_len=%d)", size, seq_len)
        latent = rng.standard_normal(
            (1, self.cfg.latent_channels, latent_h, latent_w),
            dtype=np.float32,
        ).astype(np.float16)
        latent = scheduler.scale_initial_noise(latent)

        log.info("Diffusion loop: %d steps", steps)
        for i, t in enumerate(scheduler.timesteps):
            log.info("  step %d/%d (sigma=%.4f)", i + 1, steps, scheduler.sigmas[i])
            seq = _patchify(latent, self.cfg.patch_size)
            v = self._run_transformer(
                seq.astype(np.float16),
                text_embeds,
                np.array([t], dtype=np.float16),
            )
            v_grid = _unpatchify(
                v, self.cfg.patch_size, latent_h, latent_w, self.cfg.latent_channels
            )
            latent = scheduler.step(v_grid.astype(np.float32), latent.astype(np.float32), i).astype(np.float16)

        log.info("Decoding VAE")
        image = self._decode_vae(latent)
        # (-1, 1) → (0, 255) uint8 HxWx3
        image = ((image[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return image

    def dry_run(self, size: int = 512) -> None:
        """Run one transformer pass with random noise; verify no shape errors."""
        latent_h = size // self.cfg.vae_scale_factor
        latent_w = size // self.cfg.vae_scale_factor
        seq_len = (latent_h // self.cfg.patch_size) * (latent_w // self.cfg.patch_size)
        rng = np.random.default_rng(0)
        seq = rng.standard_normal(
            (1, seq_len, self.cfg.latent_channels * self.cfg.patch_size ** 2),
            dtype=np.float32,
        ).astype(np.float16)
        text = rng.standard_normal((1, self.cfg.text_max_length, self.cfg.hidden_size),
                                   dtype=np.float32).astype(np.float16)
        ts = np.array([500.0], dtype=np.float16)
        log.info("Dry-run transformer (seq_len=%d)", seq_len)
        v = self._run_transformer(seq, text, ts)
        if v.shape != seq.shape:
            raise NpuUnavailable(
                f"Dry-run transformer output shape {v.shape} != input {seq.shape}"
            )
        log.info("Dry-run OK")
