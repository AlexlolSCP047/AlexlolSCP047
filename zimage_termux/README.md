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

## One-shot install (on the phone)

After you have a `zimage_qnn_v79.tar.zst` published somewhere reachable by
HTTPS (see [Compile section](#compile-qnn-artifacts-on-a-workstation)), the
entire install is a single command in Termux:

```bash
pkg install -y curl
export ZIMAGE_ARTIFACTS_URL="https://your.host/zimage_qnn_v79.tar.zst"
curl -fL https://raw.githubusercontent.com/AlexlolSCP047/AlexlolSCP047/claude/termux-npu-image-generator-AXOM5/zimage_termux/Zimage-install.sh | bash
```

That one script clones the repo, installs Python packages, symlinks
`/vendor/lib64/libQnn*.so` into `$PREFIX/lib`, sets `ADSP_LIBRARY_PATH`,
downloads + extracts the QNN context binaries, and runs `Zimage --check`
to confirm the Hexagon HTP is healthy. After it finishes, the only command
you need is:

```bash
Zimage a calico cat on a windowsill at sunrise
```

Quotes are optional — every word after `Zimage` is concatenated into the
prompt. Output goes to `zimage_out.png` in the current directory.

### Manual install (if you prefer)

```bash
pkg install -y git
git clone <this-repo>
cd zimage_termux
bash setup.sh
source $PREFIX/etc/profile.d/zimage_qnn.sh
```

If `pip install onnxruntime-qnn` fails (Termux's Bionic-with-prefix layout
sometimes confuses the wheel's `dlopen`), use the manual route from
<https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html>:
download the arm64-v8a wheel from a recent release and
`pip install <file>.whl` directly.

### Why we don't `pip install numpy`

Termux Python 3.13 has no prebuilt numpy wheel, so plain `pip install numpy`
falls back to a source build that pulls in cmake/ninja and OOM-kills `make`
on the phone. `setup.sh` installs `python-numpy` and `python-pillow` from
Termux's own binary repo (which has them prebuilt for the same Python ABI)
and only pip-installs the pure-Python or Rust-buildable extras
(`huggingface_hub`, `tokenizers`). `tokenizers` builds from source on first
run — Rust is installed by `setup.sh` and the build takes 5–10 minutes
single-threaded; subsequent runs are instant.

## Quality preset (locked, not user-editable)

`Zimage` always runs with these baked-in settings:

- 8 diffusion steps (Z-Image-Turbo's NFE sweet spot)
- 1024 px output (override with `--size`)
- Mild CFG (guidance_scale = 2.0) using a fixed negative prompt that steers
  the result away from common failure modes: low quality, jpeg artifacts,
  bad anatomy, extra/missing limbs, watermarks, text/logos, oversaturation,
  draft sketches, and so on.

The negative prompt is a constant in
[`zimage/quality.py`](./zimage/quality.py). It is intentionally not
exposed via any flag — change it by editing that file and reinstalling.
Because CFG runs the transformer twice per step, image latency roughly
doubles vs. a no-negative run. Pass `--fast` to disable CFG when you want
the unboosted speed.

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
# Just the prompt — quotes optional:
Zimage a calico cat on a windowsill at sunrise

# With a specific output path and seed:
Zimage --out cat.png --seed 42 a calico cat on a windowsill at sunrise

# Speed mode (no CFG, no negative prompt):
Zimage --fast a calico cat at sunrise

# Diagnostics:
Zimage --check
Zimage --dry-run
```

Expected latency on Snapdragon 8 Elite Gen 5: **30–80 s per 1024×1024 image**
with the locked CFG=2.0 quality preset (transformer runs twice per step).
With `--fast` (CFG off) it drops to about **15–40 s**.

## CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `prompt` (positional, variadic) | — | All words after `Zimage` are joined into the prompt. |
| `--size` | `1024` | Must be a multiple of `vae_scale_factor * patch_size` (typically 16). |
| `--seed` | random | Integer for reproducible noise. |
| `--out` | `zimage_out.png` | Output PNG path. |
| `--fast` | off | Disable CFG (skip negative prompt) for ~2× speed. |
| `--check` | — | Run NPU gate + artifact discovery; exit 0 on success. |
| `--dry-run` | — | Load all sessions and run one step on noise. |
| `--verbose` / `-v` | — | Log each diffusion step. |

Steps, guidance scale, and negative prompt are **not** flags — they're
locked in [`zimage/quality.py`](./zimage/quality.py).

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
