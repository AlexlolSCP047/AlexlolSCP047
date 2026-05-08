# Off-device compile toolkit

These scripts run on a **Linux x86_64 workstation**, not on the phone. They turn
the published Z-Image-Turbo PyTorch checkpoint into Hexagon-ready QNN context
binaries that the on-device runtime loads.

## Prerequisites (workstation)

- Linux x86_64 (Ubuntu 22.04 or newer recommended).
- Python 3.11 with `torch`, `diffusers`, `huggingface_hub`, `onnx`,
  `onnxruntime`, `numpy`.
- **Qualcomm AI Engine Direct SDK (QAIRT) ≥ 2.28**, downloaded from
  <https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk>.
  After install, source the env script:

  ```bash
  source ${QAIRT_ROOT}/bin/envsetup.sh
  export QNN_SDK_ROOT="${QAIRT_ROOT}"
  ```

  This puts `qnn-onnx-converter`, `qnn-model-lib-generator`, and
  `qnn-context-binary-generator` on `PATH`.
- ~30 GB free disk for intermediate artifacts.
- ~30 GB free RAM for ONNX export of the 6B DiT.

## Pipeline

```bash
cd compile

# 1. Export the Tongyi-MAI/Z-Image-Turbo checkpoint to ONNX, partitioned
#    per DiT block. Outputs to ./onnx_out/.
python export_onnx.py --out-dir ./onnx_out --size 1024

# 2. (Optional but recommended) W8A16 post-training quantization of the
#    DiT blocks. Requires calibration captures — see "Calibration" below.
python quantize.py --in-dir ./onnx_out --out-dir ./onnx_quant

# 3. Compile each ONNX into a QNN context binary for HTP arch v79.
HTP_ARCH=v79 ./compile_qnn.sh ./onnx_quant ./qnn_out

# 4. Bundle for transfer to the phone.
./pack_artifacts.sh ./qnn_out zimage_qnn_v79.tar.zst
```

Then on the phone:

```bash
mkdir -p ~/.cache/zimage/qnn
tar -I zstd -xf /sdcard/Download/zimage_qnn_v79.tar.zst -C ~/.cache/zimage/qnn/
zimage --check
```

## Calibration

`quantize.py` needs activation captures from the FP16 reference graph to set
per-tensor scales. The simplest path:

1. Run the reference `ZImagePipeline` (PyTorch FP16) in a script that hooks
   each DiT block's input and saves it to
   `onnx_out/transformer_block_NN.calib/<idx>.npz` for the prompts in
   `quantize.py::CALIBRATION_PROMPTS`.
2. 32 prompts × 8 steps × N blocks ≈ a few GB on disk.

If you skip quantization (step 2), the DiT compiles in FP16 — about 2× more
HTP weight memory and ~1.5–2× slower. Acceptable for first-light testing,
required to be replaced by W8A16 for a 16 GB phone with comfortable headroom.

## HTP arch

`HTP_ARCH=v79` matches Snapdragon 8 Elite Gen 5. Verify on the actual device:

```bash
ls /vendor/lib64/libQnnHtpV*Stub.so
```

The numeric portion is the arch — pass it as `HTP_ARCH=v{NN}`.

## Notes

- Z-Image is **not on Qualcomm AI Hub** as of 2026-05; you cannot download
  pre-compiled binaries. This pipeline is currently the only path.
- The on-device runtime refuses to start without these artifacts. By
  design, there is no CPU fallback.
