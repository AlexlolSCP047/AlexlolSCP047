#!/usr/bin/env bash
# Bundle the QNN context binaries + tokenizer + scheduler config into a tar
# the user copies to the phone and extracts under ~/.cache/zimage/qnn/.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <qnn_out_dir> [output.tar.zst]" >&2
    exit 1
fi

QNN_DIR="$1"
OUT="${2:-zimage_qnn_v79.tar.zst}"

if ! command -v zstd >/dev/null 2>&1; then
    echo "ERROR: zstd not installed on workstation." >&2
    exit 2
fi

cd "$(dirname "${QNN_DIR}")"
tar --use-compress-program "zstd -19 -T0" \
    -cf "${OUT}" \
    -C "${QNN_DIR}" .

echo "Wrote ${OUT}"
echo "Copy to phone, then:  tar -I zstd -xf ${OUT} -C ~/.cache/zimage/qnn/"
