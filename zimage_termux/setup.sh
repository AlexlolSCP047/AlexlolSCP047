#!/data/data/com.termux/files/usr/bin/env bash
# Termux-side installer for zimage-termux.
# Run inside Termux on the Samsung S26 Ultra (no root needed).
set -euo pipefail

if [[ -z "${PREFIX:-}" || ! -d "${PREFIX}" ]]; then
    echo "ERROR: \$PREFIX is not set or invalid. This script must run inside Termux." >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/6] Updating Termux package index"
pkg update -y

echo "[2/6] Installing Termux base toolchain"
# Native deps via Termux's binary packages — avoids pip building cmake/ninja
# from source (which OOMs on phones). python-numpy and python-pillow are
# prebuilt for Termux's Python; we deliberately do NOT pip-install them.
pkg install -y \
    python python-pip git clang make pkg-config \
    libjpeg-turbo libpng zstd binutils \
    python-numpy python-pillow rust

echo "[3/6] Installing pure-Python dependencies via pip"
# --no-build-isolation keeps pip from rebuilding cmake/ninja. We pre-install
# wheel + setuptools so the build backends already exist in the runtime env.
python -m pip install --upgrade --no-input wheel setuptools

# Pre-install Rust build backends so tokenizers (and any other Rust-based
# package we touch) can compile under --no-build-isolation. maturin pulls a
# small prebuilt wheel for arm64; setuptools-rust is pure Python.
python -m pip install --no-input --no-build-isolation \
    "maturin>=1.5" "setuptools-rust>=1.7"

# tokenizers needs Rust (installed above). Build is single-threaded by
# default; expect 5–10 minutes on first run.
if ! python -c "import tokenizers" >/dev/null 2>&1; then
    echo "  Building tokenizers from source (uses Rust, ~5-10 min on first run)..."
    CARGO_BUILD_JOBS=1 python -m pip install --no-input --no-build-isolation \
        "tokenizers>=0.20"
fi

# Note: we deliberately do NOT install huggingface_hub here. It's only used
# by compile/export_onnx.py on the workstation. Newer huggingface_hub
# pulls in hf-xet (Rust + maturin) which is unnecessary friction on Termux,
# and the runtime never downloads from the Hub — tokenizer.json ships in
# the QNN artifact tar.

echo "[4/6] Installing onnxruntime-qnn (best-effort)"
# Termux pip wheels for onnxruntime-qnn are best-effort. If pip fails, the
# README documents the manual route: fetch the arm64-v8a wheel from the
# Qualcomm developer portal and `pip install <file>.whl` directly.
if ! python -m pip install --no-input --no-build-isolation "onnxruntime-qnn" 2>/dev/null; then
    echo "  WARNING: pip install onnxruntime-qnn failed."
    echo "  Follow the manual install path in README.md before running 'Zimage --check'."
fi

echo "[5/6] Wiring Hexagon vendor libraries into Termux prefix"
bash "${HERE}/vendor_shim.sh"

echo "[6/6] Installing the zimage package (editable)"
python -m pip install --no-input --no-build-isolation -e "${HERE}"

echo
echo "Setup complete. Next steps:"
echo "  1. Source the QNN profile:  source \$PREFIX/etc/profile.d/zimage_qnn.sh"
echo "  2. Make sure ~/.cache/zimage/qnn/ has the QNN context binaries"
echo "     (compile on a workstation per compile/README.md, or set"
echo "      ZIMAGE_ARTIFACTS_URL and re-run Zimage-install.sh)"
echo "  3. Run:  Zimage --check"
