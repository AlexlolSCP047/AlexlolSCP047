#!/usr/bin/env bash
# Bundle the QNN runtime libraries from a QAIRT SDK install into a tar
# the user extracts into Termux's $PREFIX/lib on the phone.
#
# Stock Samsung firmware does NOT ship the QNN libraries — they come from
# Qualcomm's QAIRT SDK. This script copies the aarch64-android variant
# of the runtime libs into a small tar.zst that the on-device installer
# extracts (~70 MB compressed, ~200 MB uncompressed).
#
# Prerequisites (workstation):
#   - QAIRT SDK >= 2.28 installed; QNN_SDK_ROOT exported.
#   - zstd available.
#
# Usage:
#   QNN_SDK_ROOT=/opt/qairt/2.28.0 HTP_ARCH=v79 \
#       ./bundle_qnn_runtime.sh ./qnn_runtime_v79.tar.zst
set -euo pipefail

OUT="${1:-qnn_runtime.tar.zst}"
HTP_ARCH="${HTP_ARCH:-v79}"   # Snapdragon 8 Elite Gen 5 expected; verify.
ARCH_NUM="${HTP_ARCH#v}"

if [[ -z "${QNN_SDK_ROOT:-}" || ! -d "${QNN_SDK_ROOT}" ]]; then
    echo "ERROR: QNN_SDK_ROOT is not set or invalid." >&2
    exit 2
fi

LIB_DIR="${QNN_SDK_ROOT}/lib/aarch64-android"
DSP_DIR="${QNN_SDK_ROOT}/lib/hexagon-${HTP_ARCH}/unsigned"

if [[ ! -d "${LIB_DIR}" ]]; then
    echo "ERROR: ${LIB_DIR} not found. Wrong QNN_SDK_ROOT or layout?" >&2
    exit 3
fi
if [[ ! -d "${DSP_DIR}" ]]; then
    echo "ERROR: ${DSP_DIR} not found. Wrong HTP_ARCH (${HTP_ARCH}) or layout?" >&2
    exit 3
fi
if ! command -v zstd >/dev/null 2>&1; then
    echo "ERROR: zstd not installed." >&2
    exit 4
fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# 1. CPU/system-side libs (loaded by onnxruntime in Termux)
mkdir -p "${WORK}/lib"
cp "${LIB_DIR}/libQnnHtp.so"                 "${WORK}/lib/"
cp "${LIB_DIR}/libQnnSystem.so"              "${WORK}/lib/"
cp "${LIB_DIR}/libQnnHtpPrepare.so"          "${WORK}/lib/" 2>/dev/null || true
cp "${LIB_DIR}/libQnnHtpV${ARCH_NUM}Stub.so" "${WORK}/lib/"

# 2. cDSP-side skel — QNN's HTP backend dlopens this from ADSP_LIBRARY_PATH;
#    Termux's vendor_shim.sh points ADSP_LIBRARY_PATH at /vendor/dsp/cdsp etc.
#    On stock firmware those locations are read-only; we ship the skel
#    alongside the stub and rely on the user's stock cDSP image. If your
#    device's cDSP image already exposes the skel, you can omit this step.
cp "${DSP_DIR}/libQnnHtpV${ARCH_NUM}Skel.so" "${WORK}/lib/"

# 3. Manifest so the user knows what they have.
cat >"${WORK}/MANIFEST" <<MANIFEST
QNN runtime bundle
  htp_arch     : ${HTP_ARCH}
  qnn_sdk_root : ${QNN_SDK_ROOT}
  built_on     : $(date -u +%FT%TZ)
  files        :
$(cd "${WORK}/lib" && ls -la | sed 's/^/    /')
MANIFEST

cd "${WORK}"
tar --use-compress-program "zstd -19 -T0" -cf "${OLDPWD}/${OUT}" .
echo "Wrote ${OUT}"
echo "On the phone, extract into Termux:"
echo "  tar -I zstd -xf ${OUT} -C \$PREFIX/"
