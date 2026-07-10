#!/usr/bin/env bash
#
# Set up an AWS Trainium (trn1/trn2) instance for this repo's evaluation
# scripts, starting from a bare Ubuntu AMI. Idempotent — safe to re-run.
#
#   bash setup_trainium.sh
#
# Afterwards:
#   source ~/neuron_venv/bin/activate
#   export PJRT_DEVICE=NEURON
#   python evaluate/evaluate_crosslingual_consistency.py --device xla ...
#
# Verified on trn2.3xlarge, Ubuntu 26.04, kernel 7.0.0-1006-aws (2026-07-10).
# On an official Neuron DLAMI (Ubuntu 22.04/24.04) most steps are no-ops and
# none of the compat fixes below should trigger.
#
# Known quirks this script handles (all hit on Ubuntu 26.04):
#  1. apt-key is gone            -> signed-by keyring for the Neuron apt repo
#  2. No 26.04 dist in the repo  -> uses noble (24.04) packages
#  3. aws-neuronx-dkms fails on kernel >= 7.0: mm_get_unmapped_area() lost
#     its `mm` argument          -> patches the driver source and rebuilds
#  4. neuronx-cc's walrus_driver needs libarchive.so.13 and libxml2.so.2,
#     which 26.04 no longer ships -> libarchive13t64 + noble libxml2 into
#     /opt/neuron-compat-libs
#  5. torch_xla needs libpython3.11.so, which uv's standalone python keeps
#     out of the default search path -> LD_LIBRARY_PATH in venv activate
#
# One more gotcha (not scriptable): Neuron caches FAILED compilations in
# /var/tmp/neuron-compile-cache. If a compile fails due to a missing library
# and you fix it, delete that directory or the old failure replays.

set -euo pipefail

VENV="$HOME/neuron_venv"
PYTHON_VERSION=3.11
COMPAT_DIR=/opt/neuron-compat-libs

log() { echo -e "\n==> $*"; }

# ---------------------------------------------------------------- apt repo
log "Neuron apt repository (noble packages, signed-by keyring)"
if [ ! -f /etc/apt/keyrings/neuron.gpg ]; then
    sudo mkdir -p /etc/apt/keyrings
    wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
        | gpg --dearmor | sudo tee /etc/apt/keyrings/neuron.gpg > /dev/null
fi
echo "deb [signed-by=/etc/apt/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com noble main" \
    | sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null
sudo apt-get update -y

# ------------------------------------------------------- driver + runtime
log "Neuron driver (DKMS), runtime, collectives, tools"
sudo apt-get install -y dkms "linux-headers-$(uname -r)" \
    aws-neuronx-collectives aws-neuronx-runtime-lib aws-neuronx-tools libarchive13t64

if ! sudo apt-get install -y aws-neuronx-dkms; then
    log "aws-neuronx-dkms build failed — applying kernel>=7.0 mm_get_unmapped_area patch"
    SRC=$(ls -d /usr/src/aws-neuronx-* | sort -V | tail -1)
    if ! grep -q "KERNEL_VERSION(7, 0, 0)" "$SRC/neuron_mmap.h"; then
        sudo perl -0pi -e 's{#if (\(!defined\(RHEL_RELEASE_CODE\) && \(LINUX_VERSION_CODE >= KERNEL_VERSION\(6, 10, 0\)\)\).*?\n)(#define nmmap_kern_get_unmapped_area\(filep, addr, len, pgoff, flags\) \\\n\tmm_get_unmapped_area\(current->mm, filep, addr, len, pgoff, flags\))}
{#if (!defined(RHEL_RELEASE_CODE) && (LINUX_VERSION_CODE >= KERNEL_VERSION(7, 0, 0)))
/* Linux 7.0 dropped the mm argument from mm_get_unmapped_area() */
#define nmmap_kern_get_unmapped_area(filep, addr, len, pgoff, flags) \\
\tmm_get_unmapped_area(filep, addr, len, pgoff, flags)
#elif $1$2}s' "$SRC/neuron_mmap.h"
    fi
    VER=$(basename "$SRC" | sed 's/aws-neuronx-//')
    sudo dkms build "aws-neuronx/$VER"
    sudo dkms install "aws-neuronx/$VER" || true
    sudo modprobe neuron
    sudo dpkg --configure -a || true
fi
sudo modprobe neuron || true
ls /dev/neuron* > /dev/null  # fail loudly if the driver is not loaded
/opt/aws/neuron/bin/neuron-ls

# ------------------------------------------------- libxml2.so.2 compat lib
if ! ldconfig -p | grep -q "libxml2.so.2 "; then
    log "libxml2.so.2 missing (needed by neuronx-cc) — installing noble build into $COMPAT_DIR"
    sudo mkdir -p "$COMPAT_DIR"
    TMP=$(mktemp -d)
    DEB=$(curl -s "http://archive.ubuntu.com/ubuntu/pool/main/libx/libxml2/" \
          | grep -oE 'libxml2_2\.1[0-9][^"]*_amd64\.deb' | sort -uV | tail -1)
    curl -so "$TMP/libxml2.deb" "http://archive.ubuntu.com/ubuntu/pool/main/libx/libxml2/$DEB"
    dpkg-deb -x "$TMP/libxml2.deb" "$TMP/x"
    sudo cp -a "$TMP"/x/usr/lib/x86_64-linux-gnu/libxml2.so.2* "$COMPAT_DIR/"
    rm -rf "$TMP"
fi

# ------------------------------------------------------------ python venv
log "Python $PYTHON_VERSION venv with torch-neuronx"
if ! command -v uv > /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
[ -d "$VENV" ] || uv venv "$VENV" --python "$PYTHON_VERSION"

# torch_xla links against libpython; uv's standalone cpython keeps it here:
PYLIB=$(dirname "$(find "$HOME/.local/share/uv/python" -name "libpython${PYTHON_VERSION}.so.1.0" | head -1)")
for LINE in \
    "export LD_LIBRARY_PATH=$PYLIB:\$LD_LIBRARY_PATH" \
    "export LD_LIBRARY_PATH=$COMPAT_DIR:\$LD_LIBRARY_PATH" \
    "export PJRT_DEVICE=NEURON" \
    "export PATH=\$PATH:/opt/aws/neuron/bin"; do
    grep -qF "$LINE" "$VENV/bin/activate" || echo "$LINE" >> "$VENV/bin/activate"
done

# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install --index-strategy unsafe-best-match \
    --extra-index-url=https://pip.repos.neuron.amazonaws.com \
    torch-neuronx neuronx-cc transformers datasets sentence-transformers accelerate

# ---------------------------------------------------------------- verify
log "Verifying installation"
python - <<'EOF'
import torch, torch_neuronx, transformers
import torch_xla.core.xla_model as xm
dev = xm.xla_device()
x = torch.ones(2, 2, device=dev)
assert (x + x).sum().item() == 8.0
print(f"OK: torch {torch.__version__}, transformers {transformers.__version__}, XLA device {dev}")
EOF

log "Done. Use with:  source $VENV/bin/activate   (PJRT_DEVICE is set by activate)"
