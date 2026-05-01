#!/data/data/com.termux/files/usr/bin/env bash
# One-shot installer for the Zimage runner on Termux (Samsung S26 Ultra).
#
# After running this once, the only command you need is:
#     Zimage <prompt words ...>
#
# What this script does, in order:
#   1. Verifies it is running inside Termux.
#   2. Clones (or updates) the repository into $ZIMAGE_HOME (default: ~/zimage-termux).
#   3. Runs setup.sh, which installs Python, the QNN ORT wheel, and symlinks
#      the Hexagon vendor libraries.
#   4. Sources the QNN profile so ADSP_LIBRARY_PATH is set.
#   5. Downloads pre-compiled QNN context binaries from $ZIMAGE_ARTIFACTS_URL
#      and extracts them to ~/.cache/zimage/qnn/.
#      (Set ZIMAGE_ARTIFACTS_URL=<https://.../zimage_qnn_v79.tar.zst> first;
#      see compile/README.md for how to produce that tar on a workstation.
#      Z-Image is not on Qualcomm AI Hub, so somebody has to compile it once.)
#   6. Runs `Zimage --check` to verify HTP gate + artifacts.
#
# One-line bootstrap (after pasting this script into Termux, or fetching it):
#     export ZIMAGE_ARTIFACTS_URL="https://.../zimage_qnn_v79.tar.zst"
#     bash Zimage-install.sh
set -euo pipefail

REPO_URL="${ZIMAGE_REPO_URL:-https://github.com/AlexlolSCP047/AlexlolSCP047.git}"
BRANCH="${ZIMAGE_BRANCH:-claude/termux-npu-image-generator-AXOM5}"
INSTALL_DIR="${ZIMAGE_HOME:-${HOME}/zimage-termux}"
CACHE_DIR="${HOME}/.cache/zimage/qnn"
ARTIFACTS_URL="${ZIMAGE_ARTIFACTS_URL:-}"

log()   { printf '\033[1;36m[Zimage-install]\033[0m %s\n' "$*"; }
fatal() { printf '\033[1;31m[Zimage-install ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# 1. Termux sanity check
if [[ -z "${PREFIX:-}" || ! -d "${PREFIX}" || "${PREFIX}" != /data/data/com.termux/* ]]; then
    fatal "This installer must run inside Termux (PREFIX=${PREFIX:-unset})."
fi

# 2. Ensure git, then clone or update
log "Bootstrapping git"
pkg install -y git curl ca-certificates >/dev/null

if [[ -d "${INSTALL_DIR}/.git" ]]; then
    log "Updating existing checkout in ${INSTALL_DIR}"
    git -C "${INSTALL_DIR}" fetch --depth=1 origin "${BRANCH}"
    git -C "${INSTALL_DIR}" checkout "${BRANCH}"
    git -C "${INSTALL_DIR}" reset --hard "origin/${BRANCH}"
else
    log "Cloning ${REPO_URL} (${BRANCH}) into ${INSTALL_DIR}"
    git clone --depth=1 --branch "${BRANCH}" "${REPO_URL}" "${INSTALL_DIR}"
fi

ZT_DIR="${INSTALL_DIR}/zimage_termux"
[[ -d "${ZT_DIR}" ]] || fatal "${ZT_DIR} not found in checkout."

# 3. Run setup.sh
log "Running setup.sh"
bash "${ZT_DIR}/setup.sh"

# 4. Source QNN profile so ADSP_LIBRARY_PATH is exported in this shell
PROFILE="${PREFIX}/etc/profile.d/zimage_qnn.sh"
if [[ -f "${PROFILE}" ]]; then
    # shellcheck disable=SC1090
    source "${PROFILE}"
else
    fatal "Expected ${PROFILE} after setup.sh — did vendor_shim.sh succeed?"
fi

# 5. Download QNN artifacts
mkdir -p "${CACHE_DIR}"
if [[ -n "${ARTIFACTS_URL}" ]]; then
    TAR_FILE="${CACHE_DIR}/zimage_qnn_artifacts.tar.zst"
    log "Downloading QNN artifacts from ${ARTIFACTS_URL}"
    pkg install -y zstd tar >/dev/null
    curl -fL --retry 4 --retry-delay 2 -o "${TAR_FILE}" "${ARTIFACTS_URL}"
    log "Extracting into ${CACHE_DIR}"
    tar -I zstd -xf "${TAR_FILE}" -C "${CACHE_DIR}"
    rm -f "${TAR_FILE}"
else
    log "ZIMAGE_ARTIFACTS_URL not set — skipping artifact download."
    log "Compile context binaries on a workstation (compile/README.md) and"
    log "extract them into ${CACHE_DIR}, then re-run: Zimage --check"
fi

# 6. Check
log "Running 'Zimage --check'"
if Zimage --check; then
    log "Install complete. Try:  Zimage a calico cat on a windowsill at sunrise"
else
    fatal "Zimage --check failed. See messages above."
fi
