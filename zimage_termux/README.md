# zimage-termux

Run **Tongyi-MAI/Z-Image-Turbo** locally on a **Samsung Galaxy S26 Ultra** from
**Termux**, using the on-board **Hexagon NPU** (Qualcomm HTP) as the only
execution backend.

> **NPU-only / fail-loud.** This runner refuses to start unless the QNN
> ExecutionProvider is initialized on the Hexagon HTP. There is no CPU or GPU
> fallback by design.

## Architecture in one paragraph

Z-Image-Turbo is a 6B-parameter Single-Stream Diffusion Transformer. It is
not published in a Hexagon-ready form, so a workstation runs the
[`compile/`](./compile) toolkit to export PyTorch → partitioned ONNX → QNN
context binaries (one `.qnn.bin` per DiT block, plus text encoder and VAE
decoder). Those artifacts are copied to `~/.cache/zimage/qnn/` on the phone.
At runtime, the [`zimage/`](./zimage) package loads them via
`onnxruntime-qnn`, runs an 8-step flow-matching loop, and decodes a PNG —
keeping memory under 16 GB by lazy-loading and freeing each ORT session.

## Installation (on the phone)

```bash
pkg install -y git
git clone <this-repo>
cd zimage_termux
bash setup.sh
source $PREFIX/etc/profile.d/zimage_qnn.sh
```

`setup.sh` installs Python packages, symlinks `/vendor/lib64/libQnn*.so`
into `$PREFIX/lib`, and sets `ADSP_LIBRARY_PATH` so the Hexagon stub finds
its skel.

If `pip install onnxruntime-qnn` fails (Termux's Bionic-with-prefix layout
sometimes confuses the wheel's `dlopen`), use the manual route from
<https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html>:
download the arm64-v8a wheel from a recent release and
`pip install <file>.whl` directly.

## Compile QNN artifacts (on a workstation)

See [`compile/README.md`](./compile/README.md). One-paragraph summary:

```bash
source ${QAIRT_ROOT}/bin/envsetup.sh
cd compile
python export_onnx.py --out-dir ./onnx_out --size 1024
python quantize.py    --in-dir  ./onnx_out --out-dir ./onnx_quant
HTP_ARCH=v79 ./compile_qnn.sh ./onnx_quant ./qnn_out
./pack_artifacts.sh ./qnn_out zimage_qnn_v79.tar.zst
```

Copy the tar to the phone, then:

```bash
mkdir -p ~/.cache/zimage/qnn
tar -I zstd -xf /sdcard/Download/zimage_qnn_v79.tar.zst -C ~/.cache/zimage/qnn/
```

## Running

```bash
# Verify HTP gate + artifact presence:
zimage --check

# Sanity-check shapes without waiting 8 steps:
zimage --dry-run

# Generate:
zimage "a photo of a calico cat on a windowsill" \
    --steps 8 --size 1024 --seed 42 --out cat.png
```

Expected latency on Snapdragon 8 Elite Gen 5: **15–40 s per 1024×1024 image**
at 8 NFEs with W8A16 weights.

## CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `prompt` (positional) | — | Required for generation. |
| `--steps` | `8` | Z-Image-Turbo's NFE sweet spot; values >16 give diminishing returns. |
| `--size` | `1024` | Must be a multiple of `vae_scale_factor * patch_size` (typically 16). |
| `--seed` | random | Integer for reproducible noise. |
| `--out` | `zimage_out.png` | Output path. |
| `--cache-dir` | `~/.cache/zimage/qnn` | QNN artifact location. |
| `--check` | — | Run NPU gate + artifact discovery; exit 0 on success. |
| `--dry-run` | — | Load all sessions and run one step on noise. |
| `--verbose` / `-v` | — | Log each diffusion step. |

## Failure modes

- `NPU gate FAILED: Missing /vendor/lib64/libQnnHtp.so` — the device firmware
  does not expose Qualcomm QNN libraries. Cannot proceed; this runner does
  not implement CPU fallback.
- `QNN EP not active after init` — `onnxruntime-qnn` was unable to keep the
  session on HTP and tried to demote to CPU. The session-level
  `disable_cpu_ep_fallback=1` guard catches this and aborts. Verify
  `ADSP_LIBRARY_PATH`, then re-run `vendor_shim.sh`.
- `Missing artifacts in ~/.cache/zimage/qnn/...` — compile artifacts on a
  workstation per [`compile/README.md`](./compile/README.md).
- `--size N is not a multiple of …` — pick a multiple of 16 (or 32 for some
  configs); 512, 768, 1024, 1280, 1536 are safe.

## Licenses

- This wrapper: Apache-2.0.
- Z-Image-Turbo weights: Apache-2.0 (Tongyi-MAI / Alibaba).
- Qualcomm QAIRT SDK: governed by Qualcomm's developer license; not
  redistributed by this repo.
