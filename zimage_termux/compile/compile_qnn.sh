#!/usr/bin/env bash
# Compile every ONNX file in an input directory into a QNN context binary
# targeting the Hexagon HTP architecture of the Samsung S26 Ultra.
#
# Prerequisites on the WORKSTATION (not the phone):
#   - Qualcomm AI Engine Direct SDK (QAIRT) >= 2.28 installed and sourced:
#       source ${QAIRT_ROOT}/bin/envsetup.sh
#   - x86_64 host with Linux. The HTP cross-compiler ships in QAIRT.
#
# Outputs: one .qnn.bin per input .onnx, plus a copied pipeline_config.json
# and the scheduler/tokenizer bundle.
#
# Usage:
#   HTP_ARCH=v79 ./compile_qnn.sh ./onnx_quant ./qnn_out
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <in_dir> <out_dir>" >&2
    exit 1
fi

IN_DIR="$1"
OUT_DIR="$2"
HTP_ARCH="${HTP_ARCH:-v79}"   # Snapdragon 8 Elite Gen 5 reports v79; verify on device.

if ! command -v qnn-onnx-converter >/dev/null 2>&1; then
    echo "ERROR: qnn-onnx-converter not on PATH. Did you 'source envsetup.sh'?" >&2
    exit 2
fi
if ! command -v qnn-context-binary-generator >/dev/null 2>&1; then
    echo "ERROR: qnn-context-binary-generator not on PATH." >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

echo "Compiling for HTP arch: ${HTP_ARCH}"

for onnx in "${IN_DIR}"/*.onnx; do
    [[ -e "${onnx}" ]] || continue
    name="$(basename "${onnx}" .onnx)"
    echo "[${name}] qnn-onnx-converter"
    qnn-onnx-converter \
        --input_network "${onnx}" \
        --output_path "${WORK}/${name}.cpp" \
        --htp_archs "${HTP_ARCH}" \
        --float_bitwidth 16

    echo "[${name}] qnn-model-lib-generator"
    qnn-model-lib-generator \
        -c "${WORK}/${name}.cpp" \
        -b "${WORK}/${name}.bin" \
        -t x86_64-linux-clang \
        -o "${WORK}/lib"

    echo "[${name}] qnn-context-binary-generator"
    qnn-context-binary-generator \
        --backend "${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so" \
        --model "${WORK}/lib/x86_64-linux-clang/lib${name}.so" \
        --binary_file "${name}" \
        --output_dir "${OUT_DIR}" \
        --htp_socinfo "${HTP_ARCH}"

    # Conventionally the generator writes ${name}.bin; rename for clarity.
    if [[ -f "${OUT_DIR}/${name}.bin" ]]; then
        mv "${OUT_DIR}/${name}.bin" "${OUT_DIR}/${name}.qnn.bin"
    fi
done

# Copy pipeline + scheduler + tokenizer side-files unchanged.
for f in pipeline_config.json tokenizer.json; do
    if [[ -f "${IN_DIR}/${f}" ]]; then
        cp "${IN_DIR}/${f}" "${OUT_DIR}/${f}"
    fi
done
if [[ -d "${IN_DIR}/scheduler" ]]; then
    mkdir -p "${OUT_DIR}/scheduler"
    cp "${IN_DIR}/scheduler/config.json" "${OUT_DIR}/scheduler/config.json"
fi

echo "Done. QNN context binaries in ${OUT_DIR}"
