#!/usr/bin/env bash
# ============================================================================
# C3VDv2 论文复现入口（KFSLAM）
#
#   数据集:  /home/nefu1020210092/C3VDv2        (11 个序列, rgb/ + depth/ + pose.txt)
#   折痕图:  /home/nefu1020210092/C3VDv2_fold   (SegCol DexiNed 伪标签, <seq>/<stem>.npy)
#   输出:    /home/nefu1020210092/KFSLAM/outputs/<backbone>_<variant>_<seq>
#
# 用法:
#   source ~/activate_gs.sh
#   ./repro_c3vd_v2.sh                 # 两套 backbone × 11 序列, 顺序执行
#   ./repro_c3vd_v2.sh splatam         # 只跑 splatam
#   ./repro_c3vd_v2.sh endogslam       # 只跑 endogslam
#   GPU=1 ./repro_c3vd_v2.sh           # 指定 GPU
#   DRY=1 ./repro_c3vd_v2.sh           # 只打印生效配置
#   SEQ="c1_rectum_t2_v1 c2_sigmoid_t2_v1" ./repro_c3vd_v2.sh   # 只跑部分序列
# ============================================================================
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/home/nefu1020210092/C3VDv2}"
FOLD_ROOT="${FOLD_ROOT:-/home/nefu1020210092/C3VDv2_fold}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/nefu1020210092/KFSLAM/outputs}"
GPU="${GPU:-0}"
VARIANT="${VARIANT:-full}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

SEQUENCES=(
  c1_ascending_t2_v1
  c1_descending_t2_v2
  c1_descending_t2_v3
  c1_rectum_t2_v1
  c1_rectum_t3_v3
  c1_sigmoid1_t2_v1
  c1_sigmoid1_t3_v2
  c2_ascending_t3_v3
  c2_rectum_t3_v3
  c2_sigmoid_t2_v1
  c2_transverse2_t3_v3
)

if [[ -n "${SEQ:-}" ]]; then
  read -r -a SEQUENCES <<< "$SEQ"
fi

if [[ $# -ge 1 ]]; then
  BACKBONES=("$@")
else
  BACKBONES=(splatam endogslam)
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[error] dataset root not found: $DATASET_ROOT" >&2; exit 1
fi
if [[ ! -d "$FOLD_ROOT" ]]; then
  echo "[error] fold root not found: $FOLD_ROOT" >&2; exit 1
fi

for bb in "${BACKBONES[@]}"; do
  for seq in "${SEQUENCES[@]}"; do
    run_dir="$OUTPUT_ROOT/${bb}_${VARIANT}_${seq}"
    echo "=============================================================="
    echo "[$(date '+%F %T')] $bb / $VARIANT / $seq"
    echo "[out] $run_dir"
    echo "=============================================================="

    # 缺折痕图时提前告警（_load_fold_seg 会静默退化为全背景通道）
    missing=$(find "$FOLD_ROOT/$seq" -name '*.npy' 2>/dev/null | wc -l)
    frames=$(find "$DATASET_ROOT/$seq/rgb" -name '*.png' 2>/dev/null | wc -l)
    if [[ "$missing" -ne "$frames" ]]; then
      echo "[warn] fold maps $missing / frames $frames for $seq" >&2
    fi

    args=(run.py
      --backbone "$bb"
      --variant "$VARIANT"
      --dataset c3vd_v2
      --sequence "$seq"
      --dataset-root "$DATASET_ROOT"
      --fold-root "$FOLD_ROOT"
      --output "$run_dir"
      --gpu "$GPU")
    if [[ -n "${EXTRA_ARGS:-}" ]]; then
      # shellcheck disable=SC2206
      args+=($EXTRA_ARGS)
    fi

    if [[ "${DRY:-0}" == "1" ]]; then
      python "${args[@]}" --dry-run > /dev/null
      echo "[dry-run] ok"
      continue
    fi

    if [[ -f "$run_dir/params.npz" && "${FORCE:-0}" != "1" ]]; then
      echo "[skip] already finished: $run_dir/params.npz"
      continue
    fi

    mkdir -p "$run_dir"
    python "${args[@]}" 2>&1 | tee "$run_dir/run.log"
  done
done

echo "[$(date '+%F %T')] all runs finished"
