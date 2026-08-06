#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible LingBot-VLA 2.0 training environment for Ubuntu 24.04 x86_64.
# This script only manages the Conda/Python environment. It does not install
# system packages, CUDA drivers/toolkits, models, or datasets.

export PYTHONNOUSERSITE=1
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

ENV_NAME="lingbotvla"
RECREATE=0
RESUME=0
SKIP_SYSTEM_CHECK=0
FORCE_BUILD_FLASH_ATTN=0
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-}"
FLASH_BUILD_JOBS="${FLASH_BUILD_JOBS:-8}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

PYTHON_VERSION="3.12"
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCHAUDIO_VERSION="2.8.0"
TORCHDATA_VERSION="0.11.0"
TORCHCODEC_VERSION="0.6.0"
TRANSFORMERS_VERSION="4.57.3"
HUGGINGFACE_HUB_VERSION="0.34.3"
FLASH_ATTN_VERSION="2.8.3"
LEROBOT_VERSION="0.4.2"
MIN_DRIVER_VERSION="570.26"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() {
  printf '[lingbot-env] %s\n' "$*"
}

warn() {
  printf '[lingbot-env] WARN: %s\n' "$*" >&2
}

die() {
  printf '[lingbot-env] ERROR: %s\n' "$*" >&2
  exit 1
}

on_error() {
  local exit_code=$?
  printf '[lingbot-env] ERROR: command failed at line %s (exit %s): %s\n' \
    "${BASH_LINENO[0]:-unknown}" "${exit_code}" "${BASH_COMMAND:-unknown}" >&2
  exit "${exit_code}"
}
trap on_error ERR

usage() {
  cat <<'USAGE'
Usage: bash tools/create_train_env_ubuntu2404.sh [OPTIONS]

Create the complete LingBot-VLA 2.0 Python environment for Ubuntu 24.04:
  - Python 3.12
  - PyTorch 2.8.0 + CUDA 12.8 wheels
  - Transformers 4.57.3 / Hugging Face Hub 0.34.3
  - FlashAttention 2.8.3
  - LeRobot 0.4.2 without its incompatible dependency metadata
  - LingBot depth, MoGe, and their runtime dependencies

The script does not install OS packages, NVIDIA drivers/CUDA toolkit, models,
or datasets.

Options:
  --env-name NAME              Conda environment name (default: lingbotvla)
  --recreate                   Remove and rebuild an existing environment
  --resume                     Reconcile an existing environment in place
  --flash-attn-wheel PATH      Install a local FlashAttention wheel
  --force-build-flash-attn     Compile FlashAttention locally with nvcc
  --max-build-jobs N           Parallel jobs for a local FA build (default: 8)
  --skip-system-check          Skip Ubuntu/driver/GPU preflight checks
  -h, --help                   Show this help

Environment overrides:
  FLASH_ATTN_WHEEL=/path/to/wheel.whl
  FLASH_BUILD_JOBS=4
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

Examples:
  bash tools/create_train_env_ubuntu2404.sh --recreate
  bash tools/create_train_env_ubuntu2404.sh --resume
  bash tools/create_train_env_ubuntu2404.sh \
    --flash-attn-wheel /path/to/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || die "--env-name requires a value"
      ENV_NAME="$2"
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --flash-attn-wheel)
      [[ $# -ge 2 ]] || die "--flash-attn-wheel requires a path"
      FLASH_ATTN_WHEEL="$2"
      shift 2
      ;;
    --force-build-flash-attn)
      FORCE_BUILD_FLASH_ATTN=1
      shift
      ;;
    --max-build-jobs)
      [[ $# -ge 2 ]] || die "--max-build-jobs requires a value"
      FLASH_BUILD_JOBS="$2"
      shift 2
      ;;
    --skip-system-check)
      SKIP_SYSTEM_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "${RECREATE}" == "0" || "${RESUME}" == "0" ]] || \
  die "--recreate and --resume are mutually exclusive"
[[ "${FORCE_BUILD_FLASH_ATTN}" == "0" || -z "${FLASH_ATTN_WHEEL}" ]] || \
  die "--force-build-flash-attn and --flash-attn-wheel are mutually exclusive"
[[ "${ENV_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "invalid Conda environment name: ${ENV_NAME}"
[[ "${FLASH_BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || \
  die "--max-build-jobs must be a positive integer"

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

version_ge() {
  local actual="$1"
  local minimum="$2"
  [[ "$(printf '%s\n%s\n' "${minimum}" "${actual}" | sort -V | head -n 1)" == "${minimum}" ]]
}

preflight_system() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    die "this environment targets Linux; detected $(uname -s)"
  fi
  if [[ "$(uname -m)" != "x86_64" ]]; then
    die "this environment targets x86_64; detected $(uname -m)"
  fi

  [[ -r /etc/os-release ]] || die "/etc/os-release is not readable"
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
    die "expected Ubuntu 24.04, detected ${PRETTY_NAME:-unknown}; use --skip-system-check to override"
  fi

  require_command ldd
  local glibc_version
  glibc_version="$(ldd --version | head -n 1 | awk '{print $NF}')"
  version_ge "${glibc_version}" "2.39" || \
    die "glibc >=2.39 is required for this Ubuntu 24.04 environment; detected ${glibc_version}"

  require_command nvidia-smi
  local driver_version
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
  [[ -n "${driver_version}" ]] || die "could not read the NVIDIA driver version"
  version_ge "${driver_version}" "${MIN_DRIVER_VERSION}" || \
    die "NVIDIA driver >=${MIN_DRIVER_VERSION} is required for CUDA 12.8; detected ${driver_version}"

  local compute_caps
  if compute_caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)"; then
    while IFS= read -r compute_cap; do
      compute_cap="$(printf '%s' "${compute_cap}" | tr -d '[:space:]')"
      [[ -z "${compute_cap}" ]] && continue
      awk -v cap="${compute_cap}" 'BEGIN { exit !(cap + 0 >= 8.0) }' || \
        die "FlashAttention 2 requires compute capability >=8.0; detected ${compute_cap}"
    done <<<"${compute_caps}"
  else
    warn "nvidia-smi cannot report compute capability; the post-install CUDA check will validate the GPU"
  fi

  log "system: ${PRETTY_NAME}; glibc ${glibc_version}; NVIDIA driver ${driver_version}"
}

prepare_source_build() {
  require_command gcc
  require_command g++
  require_command nvcc

  if [[ -z "${CUDA_HOME:-}" ]]; then
    local nvcc_path
    nvcc_path="$(readlink -f "$(command -v nvcc)")"
    CUDA_HOME="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
    export CUDA_HOME
  fi
  [[ -x "${CUDA_HOME}/bin/nvcc" ]] || die "nvcc not found under CUDA_HOME=${CUDA_HOME}"

  local nvcc_release
  nvcc_release="$(${CUDA_HOME}/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
  [[ -n "${nvcc_release}" ]] || die "could not parse nvcc version"
  version_ge "${nvcc_release}" "12.8" || \
    die "local FlashAttention build requires CUDA toolkit >=12.8; detected ${nvcc_release}"

  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  log "FlashAttention source build: CUDA_HOME=${CUDA_HOME}, nvcc=${nvcc_release}, jobs=${FLASH_BUILD_JOBS}"
}

assert_python_version() {
  python - <<'PY'
import sys

assert sys.version_info[:2] == (3, 12), f"expected Python 3.12, got {sys.version}"
print("python", sys.version.split()[0])
PY
}

assert_core_stack() {
  python - <<'PY'
from importlib.metadata import version

import torch

expected = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "torchaudio": "2.8.0",
    "torchdata": "0.11.0",
    "torchcodec": "0.6.0",
    "transformers": "4.57.3",
    "tokenizers": "0.22.2",
    "huggingface-hub": "0.34.3",
    "triton": "3.4.0",
}

for package, wanted in expected.items():
    actual = version(package).split("+", 1)[0]
    assert actual == wanted, f"{package} changed: expected {wanted}, got {version(package)}"

assert torch.version.cuda == "12.8", f"expected a cu128 PyTorch wheel, got CUDA {torch.version.cuda}"
assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
for device_index in range(torch.cuda.device_count()):
    capability = torch.cuda.get_device_capability(device_index)
    assert capability >= (8, 0), (
        f"GPU {device_index} has compute capability {capability}; FlashAttention 2 requires >=(8, 0)"
    )

print(
    "core stack ok:",
    f"torch={torch.__version__}",
    f"cuda={torch.version.cuda}",
    f"gpus={torch.cuda.device_count()}",
    f"cxx11abi={torch._C._GLIBCXX_USE_CXX11_ABI}",
)
PY
}

install_flash_attention() {
  if [[ -n "${FLASH_ATTN_WHEEL}" ]]; then
    [[ -f "${FLASH_ATTN_WHEEL}" ]] || die "FlashAttention wheel not found: ${FLASH_ATTN_WHEEL}"
    log "installing local FlashAttention wheel: ${FLASH_ATTN_WHEEL}"
    python -m pip install --no-cache-dir --no-deps "${FLASH_ATTN_WHEEL}"
  elif [[ "${FORCE_BUILD_FLASH_ATTN}" == "1" ]]; then
    prepare_source_build
    log "building FlashAttention ${FLASH_ATTN_VERSION} locally"
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="${FLASH_BUILD_JOBS}" \
      python -m pip install --no-cache-dir --no-deps --no-build-isolation \
      "flash-attn==${FLASH_ATTN_VERSION}"
  else
    local cxx11_abi
    local python_tag
    local flash_wheel_url
    cxx11_abi="$(python -c 'import torch; print(str(bool(torch._C._GLIBCXX_USE_CXX11_ABI)).upper())')"
    python_tag="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
    flash_wheel_url="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.8cxx11abi${cxx11_abi}-${python_tag}-${python_tag}-linux_x86_64.whl"

    log "installing the official FlashAttention wheel for torch 2.8 / ${python_tag} / CXX11_ABI=${cxx11_abi}"
    if ! python -m pip install --no-cache-dir --no-deps "${flash_wheel_url}"; then
      die "official FlashAttention wheel installation failed; provide --flash-attn-wheel or retry with --force-build-flash-attn"
    fi
  fi

  python - <<'PY'
import flash_attn

actual = flash_attn.__version__.split("+", 1)[0]
assert actual == "2.8.3", f"expected flash-attn 2.8.3, got {flash_attn.__version__}"
print("flash_attn", flash_attn.__version__, flash_attn.__file__)
PY
}

install_local_vision_packages() {
  local lingbot_depth_dir="${REPO_ROOT}/lingbotvla/models/vla/vision_models/lingbot-depth"
  local moge_dir="${REPO_ROOT}/lingbotvla/models/vla/vision_models/MoGe"

  [[ -f "${lingbot_depth_dir}/pyproject.toml" ]] || die "missing local LingBot-Depth package"
  [[ -f "${moge_dir}/pyproject.toml" ]] || die "missing local MoGe package"

  # Their package metadata requests torch 2.6/xformers or broad unpinned
  # dependencies. --no-deps is intentional: the compatible runtime stack was
  # installed and re-pinned above.
  python -m pip install --no-cache-dir --no-deps -e "${lingbot_depth_dir}"
  python -m pip install --no-cache-dir --no-deps -e "${moge_dir}"
}

validate_environment() {
  assert_python_version
  assert_core_stack

  python - <<'PY'
import accelerate
import cv2
import flash_attn
import huggingface_hub
import mlflow
import torch
import transformers
import trimesh
import utils3d
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from mdm.model.v2 import MDMModel
from moge.model.v2 import MoGeModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

assert "flash_attention_2" in ALL_ATTENTION_FUNCTIONS
print("transformers", transformers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("accelerate", accelerate.__version__)
print("Qwen3-VL and LingBot depth imports ok")
print("GPU", torch.cuda.get_device_name(0))
PY

  if ! command -v hf >/dev/null 2>&1; then
    die "Hugging Face 'hf' CLI is missing after installation"
  fi

  # LeRobot 0.4.2 and the vendored depth packages publish dependency metadata
  # that conflicts with the versions required by LingBot-VLA 2.0. Imports and
  # core pins above are authoritative; expose pip-check output as diagnostics.
  local pip_check_output
  if ! pip_check_output="$(python -m pip check 2>&1)"; then
    warn "pip check reports expected metadata conflicts from LeRobot/depth packages:"
    printf '%s\n' "${pip_check_output}" >&2
  else
    log "pip dependency metadata check passed"
  fi
}

require_command conda
require_command git
require_command sort

if [[ "${SKIP_SYSTEM_CHECK}" == "0" ]]; then
  preflight_system
else
  warn "Ubuntu, glibc, driver, and GPU preflight checks were skipped"
fi

eval "$(conda shell.bash hook)"

ENV_EXISTS=0
if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "${ENV_NAME}"; then
  ENV_EXISTS=1
fi

if [[ "${ENV_EXISTS}" == "1" ]]; then
  if [[ "${RECREATE}" == "1" ]]; then
    log "removing existing Conda environment: ${ENV_NAME}"
    conda env remove -n "${ENV_NAME}" -y
    ENV_EXISTS=0
  elif [[ "${RESUME}" == "1" ]]; then
    log "reconciling existing Conda environment: ${ENV_NAME}"
  else
    die "Conda environment already exists: ${ENV_NAME}; use --resume or --recreate"
  fi
fi

if [[ "${ENV_EXISTS}" == "0" ]]; then
  log "creating Conda environment ${ENV_NAME} with Python ${PYTHON_VERSION}"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip -y
fi

conda activate "${ENV_NAME}"
assert_python_version

log "installing pinned Python build tools"
python -m pip install --no-cache-dir --upgrade \
  "pip==25.1.1" \
  "setuptools==80.9.0" \
  "wheel==0.45.1"

log "installing PyTorch ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
python -m pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"
python -m pip install --no-cache-dir \
  "torchdata==${TORCHDATA_VERSION}" \
  "torchcodec==${TORCHCODEC_VERSION}"

python - <<'PY'
import torch

assert torch.__version__.split("+", 1)[0] == "2.8.0", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
print("torch bootstrap ok", torch.__version__, torch.version.cuda)
PY

log "installing LingBot-VLA core requirements"
python -m pip install --no-cache-dir -r "${REPO_ROOT}/requirements.txt"

log "installing depth runtime requirements"
python -m pip install --no-cache-dir -r "${REPO_ROOT}/requirements-depth.txt"

# Broad MLflow/depth dependencies must not be allowed to move the Qwen/Torch
# stack. Re-apply the repository pins after dependency resolution.
log "restoring the pinned LingBot-VLA core stack"
python -m pip install --no-cache-dir -r "${REPO_ROOT}/requirements.txt"
python -m pip install --no-cache-dir --no-deps "numpydantic==1.9.0"
python -m pip install --no-cache-dir "huggingface-hub==${HUGGINGFACE_HUB_VERSION}"
assert_core_stack

install_flash_attention
assert_core_stack

log "installing LeRobot ${LEROBOT_VERSION} without its incompatible resolver constraints"
python -m pip install --no-cache-dir --no-deps \
  "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v${LEROBOT_VERSION}.tar.gz"

# MoGe imports utils3d at model import time. The old environment script pointed
# at a local morgbd_clean path that is not present in this repository.
log "installing the pinned MoGe utils3d runtime"
python -m pip install --no-cache-dir --no-deps \
  "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"

log "installing the local LingBot-VLA package"
python -m pip install --no-cache-dir --no-deps -e "${REPO_ROOT}"
install_local_vision_packages

# Reassert the only supported Hub/Transformers combination after every local
# package installation. In particular, do not downgrade Hub to 0.34.0.
python -m pip install --no-cache-dir --no-deps \
  "transformers==${TRANSFORMERS_VERSION}" \
  "huggingface-hub==${HUGGINGFACE_HUB_VERSION}"

log "validating the complete environment"
validate_environment

log "environment ready: ${ENV_NAME}"
log "activate with: conda activate ${ENV_NAME}"
log "models and datasets were not downloaded"
