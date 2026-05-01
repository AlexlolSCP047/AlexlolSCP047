#!/data/data/com.termux/files/usr/bin/env bash
# Symlink Qualcomm QNN vendor libraries into Termux's $PREFIX/lib so that
# onnxruntime-qnn's dlopen() finds them, and export ADSP_LIBRARY_PATH so
# the Hexagon stub can locate the matching skel on the cDSP.
#
# Termux runs as an unprivileged user — we cannot place files under /vendor;
# we only read from there. This shim creates symlinks under $PREFIX/lib that
# point at the on-device vendor libs.
set -euo pipefail

if [[ -z "${PREFIX:-}" || ! -d "${PREFIX}/lib" ]]; then
    echo "ERROR: \$PREFIX/lib not found. Run inside Termux." >&2
    exit 1
fi

VENDOR_LIB="/vendor/lib64"
DSP_DIR="/vendor/dsp"

if [[ ! -d "${VENDOR_LIB}" ]]; then
    echo "ERROR: ${VENDOR_LIB} not found. This device does not expose Qualcomm vendor libs." >&2
    exit 2
fi

shopt -s nullglob
linked=0
for lib in "${VENDOR_LIB}"/libQnn*.so "${VENDOR_LIB}"/libQnnSystem.so "${VENDOR_LIB}"/libcdsprpc.so; do
    [[ -e "${lib}" ]] || continue
    name="$(basename "${lib}")"
    ln -sf "${lib}" "${PREFIX}/lib/${name}"
    linked=$((linked + 1))
done

if [[ "${linked}" -eq 0 ]]; then
    echo "ERROR: no libQnn*.so found in ${VENDOR_LIB}." >&2
    echo "       Your device firmware does not ship Qualcomm QNN runtime libraries." >&2
    exit 3
fi

echo "Linked ${linked} Qualcomm libraries into ${PREFIX}/lib"

# Configure ADSP_LIBRARY_PATH for the Hexagon skel discovery. The skel runs
# on the cDSP and is loaded via FastRPC; the stub in libQnnHtpV*Stub.so
# searches the colon-separated paths in ADSP_LIBRARY_PATH.
PROFILE="${PREFIX}/etc/profile.d/zimage_qnn.sh"
mkdir -p "$(dirname "${PROFILE}")"
cat >"${PROFILE}" <<'PROFILE_EOF'
# Configured by zimage_termux/vendor_shim.sh — do not edit by hand.
export ADSP_LIBRARY_PATH="/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/vendor/dsp;/system/lib/rfsa/adsp;/dsp"
export LD_LIBRARY_PATH="${PREFIX}/lib:${LD_LIBRARY_PATH:-}"
PROFILE_EOF

echo "Wrote ${PROFILE}"
echo "Open a new shell or 'source ${PROFILE}' before running zimage."
