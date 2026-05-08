"""Export Tongyi-MAI/Z-Image-Turbo to ONNX, partitioned for QNN compilation.

Runs on a Linux workstation with a CUDA or CPU PyTorch install and the
`diffusers` library. NOT to be run inside Termux — this stage is entirely
off-device.

Outputs into ${OUT_DIR:-./onnx_out}/:
    text_encoder.onnx
    transformer_in.onnx
    transformer_block_NN.onnx   (one per DiT block)
    transformer_out.onnx
    vae_decoder.onnx
    pipeline_config.json        (consumed by the on-device pipeline)
    scheduler/config.json       (copied verbatim from the diffusers repo)
    tokenizer.json              (copied verbatim from the diffusers repo)

The DiT cannot be exported as a single graph: it is too large for QNN's
context-binary compiler and the Single-Stream cross-conditioning does not
trace cleanly with the default `torch.onnx.export`. We split it into a
`transformer_in` (patchify + initial embed), one ONNX file per transformer
block, and a `transformer_out` (final norm + unpatch).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch import nn

OPSET = 17
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"


class TextEncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


class TransformerInWrapper(nn.Module):
    def __init__(self, dit: nn.Module):
        super().__init__()
        self.patch_embed = dit.patch_embed
        self.text_proj = getattr(dit, "text_proj", nn.Identity())
        self.timestep_embed = dit.timestep_embed
        self.pos_embed = getattr(dit, "pos_embed", None)

    def forward(self, latent: torch.Tensor, text_embeds: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(latent)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        # Concatenate timestep along sequence; Z-Image's specific shape
        # depends on the upstream config.
        t = self.timestep_embed(timestep).unsqueeze(1)
        text = self.text_proj(text_embeds)
        return torch.cat([t, text, x], dim=1)


class TransformerBlockWrapper(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, hidden: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        return self.block(hidden, text_embeds)


class TransformerOutWrapper(nn.Module):
    def __init__(self, dit: nn.Module):
        super().__init__()
        self.norm_out = dit.norm_out
        self.proj_out = dit.proj_out

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.norm_out(hidden))


class VAEDecoderWrapper(nn.Module):
    def __init__(self, vae: nn.Module, scale_factor: float):
        super().__init__()
        self.decoder = vae.decoder
        self.scale_factor = scale_factor

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent / self.scale_factor)


def _export(module: nn.Module, args: tuple, out_path: Path, input_names: list[str], output_names: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            args,
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamic_axes=None,  # fixed shapes — required by QNN
        )
    print(f"  wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("./onnx_out"))
    ap.add_argument("--size", type=int, default=1024, help="Edge length of the image to export shapes for.")
    ap.add_argument("--text-max-length", type=int, default=256)
    ap.add_argument("--eviction-window", type=int, default=4,
                    help="Stored in pipeline_config.json — how many block sessions the runtime keeps resident.")
    args = ap.parse_args()

    print(f"Loading {MODEL_ID}")
    from diffusers import ZImagePipeline  # type: ignore

    pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    dit = pipe.transformer
    vae = pipe.vae
    text_encoder = pipe.text_encoder

    # Derive shapes from the loaded config.
    latent_channels = vae.config.latent_channels
    vae_scale = 2 ** (len(vae.config.block_out_channels) - 1)
    patch = dit.config.patch_size if hasattr(dit.config, "patch_size") else 2
    hidden = dit.config.hidden_size if hasattr(dit.config, "hidden_size") else 1536

    latent_h = args.size // vae_scale
    latent_w = args.size // vae_scale
    seq_len = (latent_h // patch) * (latent_w // patch)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting text encoder")
    ids = torch.zeros(1, args.text_max_length, dtype=torch.int32)
    mask = torch.ones(1, args.text_max_length, dtype=torch.int32)
    _export(
        TextEncoderWrapper(text_encoder),
        (ids, mask),
        args.out_dir / "text_encoder.onnx",
        ["input_ids", "attention_mask"],
        ["text_embeds"],
    )

    print("Exporting transformer_in")
    latent = torch.zeros(1, latent_channels, latent_h, latent_w, dtype=torch.float16)
    text_embeds = torch.zeros(1, args.text_max_length, hidden, dtype=torch.float16)
    ts = torch.zeros(1, dtype=torch.float16)
    _export(
        TransformerInWrapper(dit),
        (latent, text_embeds, ts),
        args.out_dir / "transformer_in.onnx",
        ["latent", "text_embeds", "timestep"],
        ["hidden"],
    )

    print(f"Exporting {len(dit.blocks)} transformer blocks")
    seq_with_cond = torch.zeros(1, 1 + args.text_max_length + seq_len, hidden, dtype=torch.float16)
    for i, block in enumerate(dit.blocks):
        _export(
            TransformerBlockWrapper(block),
            (seq_with_cond, text_embeds),
            args.out_dir / f"transformer_block_{i:02d}.onnx",
            ["hidden", "text_embeds"],
            ["hidden_out"],
        )

    print("Exporting transformer_out")
    _export(
        TransformerOutWrapper(dit),
        (seq_with_cond,),
        args.out_dir / "transformer_out.onnx",
        ["hidden"],
        ["v"],
    )

    print("Exporting VAE decoder")
    decoded_latent = torch.zeros(1, latent_channels, latent_h, latent_w, dtype=torch.float16)
    _export(
        VAEDecoderWrapper(vae, getattr(vae.config, "scaling_factor", 0.18215)),
        (decoded_latent,),
        args.out_dir / "vae_decoder.onnx",
        ["latent"],
        ["image"],
    )

    pipeline_cfg = {
        "latent_channels": int(latent_channels),
        "vae_scale_factor": int(vae_scale),
        "patch_size": int(patch),
        "hidden_size": int(hidden),
        "text_max_length": int(args.text_max_length),
        "eviction_window": int(args.eviction_window),
    }
    (args.out_dir / "pipeline_config.json").write_text(json.dumps(pipeline_cfg, indent=2))

    # Copy scheduler config and tokenizer from the cached HF snapshot.
    src_root = Path(pipe.config._name_or_path) if hasattr(pipe.config, "_name_or_path") else None
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(MODEL_ID, allow_patterns=[
        "scheduler/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer.json",
    ]))
    sched_src = snapshot / "scheduler" / "config.json"
    if sched_src.is_file():
        (args.out_dir / "scheduler").mkdir(exist_ok=True)
        shutil.copy(sched_src, args.out_dir / "scheduler" / "config.json")
    tok_candidates = [snapshot / "tokenizer" / "tokenizer.json", snapshot / "tokenizer.json"]
    for cand in tok_candidates:
        if cand.is_file():
            shutil.copy(cand, args.out_dir / "tokenizer.json")
            break

    print(f"Done. ONNX artifacts in {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
