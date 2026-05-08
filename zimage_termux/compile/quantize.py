"""W8A16 post-training quantization for the DiT blocks.

Uses ONNX Runtime's `quantize_static` with a small calibration set captured
from the FP16 reference graph. Produces per-block quantization encodings
that QAIRT consumes during context-binary generation.

Run AFTER export_onnx.py. Inputs and outputs are both ONNX files; the
content of the activations dictates the per-tensor scales.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CALIBRATION_PROMPTS = [
    "a calico cat on a windowsill at sunrise",
    "wide-angle photo of a rocky beach with crashing waves",
    "macro shot of a dewdrop on a green leaf",
    "neon-lit Tokyo street at night, cinematic",
    "oil painting of a stormy ocean and a lighthouse",
    "studio portrait of a black labrador, 85mm lens",
    "an astronaut riding a horse on the moon",
    "low-poly 3D render of a forest village",
    "watercolor sketch of cherry blossoms",
    "cyberpunk samurai standing in the rain",
    "blueprint of a vintage steam locomotive",
    "field of sunflowers under a turquoise sky",
    "isometric pixel art of a cozy bedroom",
    "stained-glass window depicting a phoenix",
    "polaroid of a 90s skateboarder mid-trick",
    "high-detail illustration of a clockwork dragon",
    "minimalist logo for a coffee shop",
    "macro shot of a circuit board with glowing traces",
    "ancient stone temple overgrown by jungle vines",
    "hot-air balloon festival at golden hour",
    "ink drawing of a koi pond",
    "snowy mountain pass with pine trees",
    "futuristic concept car on a salt flat",
    "close-up of a hummingbird in flight",
    "renaissance fresco of a bustling marketplace",
    "underwater scene with bioluminescent jellyfish",
    "vintage map of an imaginary continent",
    "cozy cabin interior with a fireplace",
    "abstract geometric pattern in pastel colors",
    "hyper-realistic eye reflecting a galaxy",
    "victorian-era automaton with brass gears",
    "skyline of a floating city above the clouds",
]


def _calibration_reader(prompts: list[str], block_path: Path) -> object:
    """A diffusers-free CalibrationDataReader that synthesizes plausible
    inputs to one transformer block from the captured FP16 activations.

    The activations are captured by running the reference FP16 ONNX graph
    once over the calibration prompts and dumping each block's input as a
    .npz file under <block_path>.calib/. quantize.py then iterates those
    captures.
    """
    from onnxruntime.quantization import CalibrationDataReader  # type: ignore

    capture_dir = block_path.with_suffix("").parent / (block_path.stem + ".calib")

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._files = sorted(capture_dir.glob("*.npz"))
            self._iter = iter(self._files)

        def get_next(self) -> dict | None:
            try:
                f = next(self._iter)
            except StopIteration:
                return None
            data = np.load(f)
            return {k: data[k] for k in data.files}

        def rewind(self) -> None:
            self._iter = iter(self._files)

    if not capture_dir.is_dir() or not list(capture_dir.glob("*.npz")):
        raise FileNotFoundError(
            f"No calibration captures in {capture_dir}. Run capture_activations.py first."
        )
    return _Reader()


def quantize_block(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=_calibration_reader(CALIBRATION_PROMPTS, src),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt16,  # A16 — required for DiT dynamic range
        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
        per_channel=True,
        reduce_range=False,
    )
    print(f"  quantized {src.name} → {dst.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("./onnx_out"))
    ap.add_argument("--out-dir", type=Path, default=Path("./onnx_quant"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    block_files = sorted(args.in_dir.glob("transformer_block_*.onnx"))
    if not block_files:
        print(f"No transformer_block_*.onnx in {args.in_dir}")
        return 1

    manifest = {"format": "W8A16-QDQ", "blocks": []}
    for src in block_files:
        dst = args.out_dir / src.name
        quantize_block(src, dst)
        manifest["blocks"].append(src.name)

    # Pass through non-block components untouched (they will be FP16-on-HTP).
    for name in ("text_encoder.onnx", "transformer_in.onnx", "transformer_out.onnx",
                 "vae_decoder.onnx", "pipeline_config.json"):
        src = args.in_dir / name
        if src.is_file():
            (args.out_dir / name).write_bytes(src.read_bytes())

    (args.out_dir / "quant_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done. Quantized DiT blocks in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
