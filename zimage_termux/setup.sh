#!/data/data/com.termux/files/usr/bin/env bash
# Termux-side installer for zimage-termux.
# Run inside Termux on the Samsung S26 Ultra (no root needed).
set -euo pipefail

if [[ -z "${PREFIX:-}" || ! -d "${PREFIX}" ]]; then
    echo "ERROR: \$PREFIX is not set or invalid. This script must run inside Termux." >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/5] Updating Termux packages"
pkg update -y
pkg install -y python python-pip git clang make pkg-config libjpeg-turbo libpng zstd binutils

echo "[2/5] Upgrading pip and installing Python dependencies"
python -m pip install --upgrade pip wheel
python -m pip install --no-input "numpy>=1.26" "pillow>=10.0" "tokenizers>=0.20" "huggingface_hub>=0.25"

echo "[3/5] Installing onnxruntime-qnn (best-effort)"
# Termux pip wheels for onnxruntime-qnn are best-effort. If this fails,
# follow README "Manual onnxruntime-qnn install" — download the arm64-v8a
# wheel from https://onnxruntime.ai/ and `pip install <file>.whl` directly.
if ! python -m pip install --no-input "onnxruntime-qnn"; then
    echo "WARNING: pip install onnxruntime-qnn failed."
    echo "         Follow the manual install path in README.md before running zimage --check."
fi

echo "[4/5] Wiring Hexagon vendor libraries into Termux prefix"
bash "${HERE}/vendor_shim.sh"

echo "[5/5] Installing the zimage package (editable)"
python -m pip install --no-input -e "${HERE}"

echo
echo "Setup complete. Next steps:"
echo "  1. Compile QNN context binaries on a workstation (see compile/README.md)"
echo "  2. Copy the resulting tar to ~/.cache/zimage/qnn/ on the phone"
echo "  3. Run:  zimage --check"
