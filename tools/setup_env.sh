#!/usr/bin/env bash
# 一键搭建 KFS⁴LAM 运行环境（Linux, 免 root）
#
#   conda create -n gs python=3.10
#   conda install -n gs pytorch=2.1.2 torchvision=0.16.2 cudatoolkit=11.8 -c pytorch
#   conda install -n gs -c nvidia cuda-nvcc cuda-cudart-dev cuda-cccl
#   conda activate gs
#   bash ./tools/setup_env.sh
#
# 说明：主光栅化器 diff-gaussian-rasterization-w-depth 固定 commit、用 SSH 拉取；
# MonoGS 消融用的独立光栅化器就地编译（与原仓库流程一致）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLONE_DIR="$ROOT/third_party/diff_gaussian_rasterization_w_depth"
REPO="git@github.com:JonathonLuiten/diff-gaussian-rasterization-w-depth.git"
REV="cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110"

echo "==> python: $(command -v python) ($(python -V 2>&1))"
python - <<'PY'
import torch
print(f"==> torch: {torch.__version__} | cuda {torch.version.cuda} | available {torch.cuda.is_available()}")
PY

echo "==> installing requirements"
python -m pip install -r "$ROOT/requirements.txt"

echo "==> fetching depth-enabled rasterizer over SSH ($REV)"
if [ ! -d "$CLONE_DIR/.git" ]; then
  GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new" git clone "$REPO" "$CLONE_DIR"
fi
git -C "$CLONE_DIR" fetch --all --tags
git -C "$CLONE_DIR" checkout "$REV"
git -C "$CLONE_DIR" submodule update --init --recursive

echo "==> building + installing diff_gaussian_rasterization"
python -m pip install --no-build-isolation "$CLONE_DIR"

echo "==> building MonoGS rasterizer (selection ablation)"
( cd "$ROOT/third_party/monogs_rasterizer" && python setup.py build_ext --inplace )

echo "==> verifying"
cd /tmp
ROOT="$ROOT" python - <<'PY'
import os, sys, importlib.util
from pathlib import Path

root = Path(os.environ["ROOT"])

import diff_gaussian_rasterization as d
print("   main rasterizer :", d.__file__)
assert hasattr(d, "GaussianRasterizer")

pkg = root / "third_party/monogs_rasterizer/diff_gaussian_rasterization/__init__.py"
spec = importlib.util.spec_from_file_location(
    "monogs_selection_raster", pkg, submodule_search_locations=[str(pkg.parent)]
)
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)
print("   monogs rasterizer:", pkg)

import open3d
print("   open3d           :", open3d.__version__)
PY

echo "==> done. Activate your env (source ~/activate_gs.sh), then: python run.py --list"
