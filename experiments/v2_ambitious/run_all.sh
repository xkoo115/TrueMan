#!/bin/bash
# TrueMan v2 实验一键启动脚本（4080Super 32GB 适配版）
#
# 关键设计：
#   - 全部产出和模型缓存重定向到 $DATA_DISK，避免撑爆 30GB 系统盘
#   - 默认 pilot scale (5 cond × 2 seed × 7d × 12h/d on Qwen-7B 4-bit)
#   - 总耗时：~20-24h；磁盘占用：~10-15GB（snapshots + captures + SAE）
#
# 用法：
#   # 服务器上一键起飞（会自动创建数据盘软链）
#   bash experiments/v2_ambitious/run_all.sh
#
#   # 自定义数据盘路径
#   DATA_DISK=/mnt/big bash experiments/v2_ambitious/run_all.sh
#
#   # Dry-run（只打印计划，不执行）
#   bash experiments/v2_ambitious/run_all.sh --dry-run
#
#   # Paper scale（4 seeds × 24h/d；约 3-4 天）
#   bash experiments/v2_ambitious/run_all.sh --paper
#
#   # 从某 stage 续跑
#   bash experiments/v2_ambitious/run_all.sh --resume-from stage2
#
#   # 查看状态
#   bash experiments/v2_ambitious/run_all.sh --status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"

# ----------------------------------------------------------------------
# 数据盘重定向（避免 30GB 系统盘被撑爆）
# ----------------------------------------------------------------------
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
RESULTS_REAL="$DATA_DISK/trueman_results"
HF_CACHE_REAL="$DATA_DISK/hf_cache"

mkdir -p "$RESULTS_REAL" "$HF_CACHE_REAL"

# HuggingFace 缓存全部走数据盘
export HF_HOME="$HF_CACHE_REAL"
export TRANSFORMERS_CACHE="$HF_CACHE_REAL/transformers"
export HF_HUB_CACHE="$HF_CACHE_REAL/hub"
export HF_DATASETS_CACHE="$HF_CACHE_REAL/datasets"
# bitsandbytes 临时文件
export TMPDIR="${TMPDIR:-$DATA_DISK/tmp}"
mkdir -p "$TMPDIR"

# results/ 软链到数据盘
RESULTS_LINK="experiments/v2_ambitious/results"
if [[ -e "$RESULTS_LINK" && ! -L "$RESULTS_LINK" ]]; then
    BACKUP="${RESULTS_LINK}.local_backup_$(date +%s)"
    echo "[disk] Backing up existing $RESULTS_LINK -> $BACKUP"
    mv "$RESULTS_LINK" "$BACKUP"
fi
if [[ ! -L "$RESULTS_LINK" ]]; then
    ln -s "$RESULTS_REAL" "$RESULTS_LINK"
    echo "[disk] symlinked $RESULTS_LINK -> $RESULTS_REAL"
fi

# ----------------------------------------------------------------------
# 参数解析
# ----------------------------------------------------------------------
DRY_RUN=""
PAPER=false
RESUME_FROM=""
STATUS_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --paper)
            PAPER=true
            shift
            ;;
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --status)
            STATUS_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--paper] [--resume-from STAGE] [--status]"
            exit 1
            ;;
    esac
done

if $STATUS_ONLY; then
    $PYTHON -m experiments.v2_ambitious.run_v2 --status
    exit 0
fi

# ----------------------------------------------------------------------
# Pilot vs Paper scale
# ----------------------------------------------------------------------
CONDITIONS="C0_trueman_full,C1_reversed,C2_scrambled,C3_frozen,C4_trivial_jaccard"
MODELS="Qwen/Qwen3-8B"

if $PAPER; then
    SEEDS="0,1,2,3"
    echo "[scope] PAPER scale: 5 cond × 4 seeds × 14d × 24h/d (~3-4d wallclock)"
    # 在 paper 模式下临时覆盖 yaml 部分参数（通过 env 传给 python）
    export TRUEMAN_OVERRIDE_DAYS=14
    export TRUEMAN_OVERRIDE_HOURS=24
else
    SEEDS="0,1"
    echo "[scope] PILOT scale: 5 cond × 2 seeds × 7d × 12h/d (~20h wallclock)"
fi

echo "============================================"
echo "TrueMan v2 Pipeline"
echo "============================================"
echo "Project root:  $PROJECT_ROOT"
echo "Python:        $PYTHON"
echo "Data disk:     $DATA_DISK"
echo "  └─ results: $RESULTS_REAL"
echo "  └─ HF cache: $HF_CACHE_REAL"
echo "Dry run:       ${DRY_RUN:-no}"
echo "Resume from:   ${RESUME_FROM:-beginning}"
echo "Conditions:    $CONDITIONS"
echo "Seeds:         $SEEDS"
echo "Models:        $MODELS"
echo "============================================"

# ----------------------------------------------------------------------
# 环境检查
# ----------------------------------------------------------------------
echo "[Pre-flight] Checking dependencies..."
$PYTHON -c "
import sys
missing = []
for mod, pkg in [('yaml','pyyaml'),('numpy','numpy'),('torch','torch'),
                 ('h5py','h5py'),('scipy','scipy'),('transformers','transformers'),
                 ('bitsandbytes','bitsandbytes')]:
    try:
        __import__(mod)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f'MISSING: {missing}')
    print(f'Install: pip install {\" \".join(missing)}')
    sys.exit(1)
import torch
if not torch.cuda.is_available():
    print('WARN: CUDA not available — will fall back to CPU (extremely slow)')
else:
    free, total = torch.cuda.mem_get_info()
    print(f'CUDA OK: {torch.cuda.get_device_name(0)} ({total/1e9:.1f}GB total, {free/1e9:.1f}GB free)')
print('All core dependencies OK')
"

mkdir -p experiments/v2_ambitious/data/probes

STAGES=("stage0" "stage1" "stage2" "stage3" "stage4" "stage5" "stage6")

run_stage() {
    local stage=$1
    echo ""
    echo "============================================"
    echo ">>> $stage <<<"
    echo "============================================"

    case $stage in
        stage0)
            $PYTHON -m experiments.v2_ambitious.run_v2 --stage stage0 $DRY_RUN --skip-env-check
            ;;
        stage1)
            $PYTHON -m experiments.v2_ambitious.run_v2 --stage stage1 \
                --conditions "$CONDITIONS" \
                --base-models "$MODELS" \
                --seeds "$SEEDS" \
                $DRY_RUN --skip-env-check
            ;;
        stage2)
            $PYTHON -m experiments.v2_ambitious.run_v2 --stage stage2 $DRY_RUN --skip-env-check
            ;;
        stage3)
            $PYTHON -m experiments.v2_ambitious.run_v2 --stage stage3 \
                --conditions "$CONDITIONS" \
                --seeds "$SEEDS" \
                $DRY_RUN --skip-env-check
            ;;
        stage4)
            # 4080S 单卡只跑 1 个底模即可（cross-model 留作 paper-scale 上的可选项）
            $PYTHON -m experiments.v2_ambitious.run_v2 --stage stage4 \
                --base-models "$MODELS" \
                --seeds "$SEEDS" \
                $DRY_RUN --skip-env-check
            ;;
        stage5)
            $PYTHON -m experiments.v2_ambitious.run_v2 --stage stage5 $DRY_RUN --skip-env-check
            ;;
        stage6)
            $PYTHON -m experiments.v2_ambitious.run_v2 --stage stage6 $DRY_RUN --skip-env-check
            ;;
    esac
}

# 确定要运行的 stages
if [[ -n "$RESUME_FROM" ]]; then
    SKIP=true
    for stage in "${STAGES[@]}"; do
        if [[ "$stage" == "$RESUME_FROM" ]]; then
            SKIP=false
        fi
        if ! $SKIP; then
            run_stage "$stage"
        fi
    done
else
    for stage in "${STAGES[@]}"; do
        run_stage "$stage"
    done
fi

echo ""
echo "============================================"
echo "Pipeline complete. Disk usage on data disk:"
echo "============================================"
du -sh "$RESULTS_REAL" "$HF_CACHE_REAL" 2>/dev/null || true

echo ""
echo "Hypothesis verdicts:"
if [[ -f "experiments/v2_ambitious/results/v2_summary.json" ]]; then
    $PYTHON -c "
import json
s = json.load(open('experiments/v2_ambitious/results/v2_summary.json'))
for h, v in s.get('hypothesis_verdicts', {}).items():
    print(f'  {h}: {v}')
"
else
    echo "  (No summary file yet — stage6 may have been skipped)"
fi

echo ""
echo "Done. Inspect: $RESULTS_REAL"
