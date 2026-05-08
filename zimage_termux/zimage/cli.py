"""`Zimage` CLI entry point.

Designed for the simplest possible invocation:

    Zimage a calico cat on a windowsill at sunrise

Everything else is baked in: 8 steps, 1024 px, locked negative prompt, mild
CFG. The only tunables are output path, seed, and image size — and admin
flags `--check` / `--dry-run` for diagnostics.

The negative prompt lives in `zimage.quality` and is intentionally not
exposed via any flag. Edit `quality.py` and reinstall to change it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .npu import NpuUnavailable, assert_htp_alive, ensure_env
from .pipeline import DEFAULT_CACHE, ZImagePipeline, discover_artifacts
from .quality import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SIZE,
    DEFAULT_STEPS,
    NEGATIVE_PROMPT,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Zimage",
        description=(
            "Z-Image-Turbo on the Hexagon NPU (Termux, NPU-only). "
            "Usage: Zimage <prompt words...>"
        ),
        epilog=(
            "Quality defaults are baked in (see zimage.quality). "
            "The negative prompt is not user-editable."
        ),
    )
    # nargs='*' so the user can write `Zimage a cat on a windowsill` without
    # quoting. We join the tokens back into a single prompt string.
    p.add_argument("prompt", nargs="*", help="Text prompt for generation.")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE,
                   help=f"Output edge length px (default: {DEFAULT_SIZE}).")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (default: random).")
    p.add_argument("--out", type=Path, default=Path("zimage_out.png"),
                   help="Output PNG path (default: zimage_out.png).")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                   help=argparse.SUPPRESS)
    p.add_argument("--check", action="store_true",
                   help="Verify HTP + artifacts and exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Load all sessions and run one step on noise.")
    p.add_argument("--fast", action="store_true",
                   help="Disable CFG (skip negative prompt) for ~2x speed.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Log every diffusion step.")
    p.add_argument("--version", action="version", version=f"Zimage {__version__}")
    return p


def _do_check(cache_dir: Path) -> int:
    try:
        info = assert_htp_alive()
    except NpuUnavailable as e:
        print(f"NPU gate FAILED: {e}", file=sys.stderr)
        return 2
    print("QNNExecutionProvider OK")
    print(f"  soc_model = {info.soc_model}")
    print(f"  htp_arch  = {info.htp_arch}")
    print(f"  backend   = {info.backend_lib}")
    print(f"  stub      = {info.stub_lib}")
    print(f"  skel      = {info.skel_lib}")
    try:
        artifacts = discover_artifacts(cache_dir)
    except FileNotFoundError as e:
        print(f"Artifact check FAILED:\n{e}", file=sys.stderr)
        return 3
    block_count = len(artifacts["transformer_blocks"])  # type: ignore[arg-type]
    print(f"Artifacts in {cache_dir}:")
    for name in sorted(artifacts):
        if name == "transformer_blocks":
            print(f"  transformer_block_*.qnn.bin ({block_count} files)")
        else:
            print(f"  {name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose or args.check or args.dry_run else logging.WARNING,
        format="%(message)s",
    )
    ensure_env()

    if args.check:
        return _do_check(args.cache_dir)

    try:
        pipe = ZImagePipeline(cache_dir=args.cache_dir)
    except (NpuUnavailable, FileNotFoundError) as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    if args.dry_run:
        try:
            pipe.dry_run(size=min(args.size, 512))
        except NpuUnavailable as e:
            print(f"Dry-run FAILED: {e}", file=sys.stderr)
            return 4
        print("Dry-run OK")
        return 0

    if not args.prompt:
        print("ERROR: prompt is required. Example: Zimage a calico cat at sunrise",
              file=sys.stderr)
        return 1

    prompt = " ".join(args.prompt).strip()

    try:
        image = pipe.generate(
            prompt,
            steps=DEFAULT_STEPS,
            size=args.size,
            seed=args.seed,
            negative_prompt="" if args.fast else NEGATIVE_PROMPT,
            guidance_scale=1.0 if args.fast else DEFAULT_GUIDANCE_SCALE,
        )
    except NpuUnavailable as e:
        print(f"Generation FAILED on NPU: {e}", file=sys.stderr)
        return 5

    from PIL import Image

    Image.fromarray(image).save(args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
