#!/bin/bash
# TrueMan v2 实验一键运行脚本
# 
# 执行顺序（严格按序）：
#   stage 0 → 构建固定刺激流 + probe 文件
#   stage 1 → 30天长时程运行 (5 cond × N seeds × M base_models)
#   stage 2 → SAE 训练 + 因果干预
#   stage 3 → Indicator battery (HOT-1/2, GWT, RPT, ΦR)
#   stage 4 → 跨模型证伪
#   stage 5 → 理论预测拟合 (FEP, PCI)
#   stage 6 → 汇总 + 假设检验
#
# 用法：
#   # 完整运行（默认: 1 model × 2 cond × 2 seed）
#   bash experiments/v2_ambitious/run_all.sh
#
#   # Dry-run（只打印计划）
#   bash experiments/v2_ambitious/run_all.sh --dry-run
#
#   # 全规模（5 cond × 4 seed × 1 model，约 200 GPU-day）
#   bash experiments/v2_ambitious/run_all.sh --full
#
#   # 从某个 stage 继续
#   bash experiments/v2_ambitious/run_all.sh --resume-from stage2
#
#   # 查看状态
#   bash experiments/v2_ambitious/run_all.sh --status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"

DRY_RUN=""
FULL=false
RESUME_FROM=""
STATUS_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --full)
            FULL=true
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
            echo "Usage: $0 [--dry-run] [--full] [--resume-from STAGE] [--status]"
            exit 1
            ;;
    esac
done

if $STATUS_ONLY; then
    $PYTHON -m experiments.v2_ambitious.run_v2 --status
    exit 0
fi

echo "============================================"
echo "TrueMan v2 Experiment Pipeline"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON"
echo "Dry run: ${DRY_RUN:-no}"
echo "Full scale: $FULL"
echo "Resume from: ${RESUME_FROM:-beginning}"
echo "============================================"

# 环境检查
echo "[Pre-flight] Checking dependencies..."
$PYTHON -c "
missing = []
for mod, pkg in [('yaml','pyyaml'),('numpy','numpy'),('torch','torch'),('h5py','h5py'),('scipy','scipy')]:
    try: __import__(mod)
    except ImportError: missing.append(pkg)
if missing:
    print(f'MISSING: {missing}')
    print(f'Install: pip install {\" \".join(missing)}')
    raise SystemExit(1)
else:
    print('All core dependencies OK')
"

# 结果目录
mkdir -p experiments/v2_ambitious/results
mkdir -p experiments/v2_ambitious/data/probes

# 构建条件参数
if $FULL; then
    CONDITIONS="C0_trueman_full,C1_reversed,C2_scrambled,C3_frozen,C4_trivial_jaccard"
    SEEDS="0,1,2,3"
    MODELS="Qwen/Qwen2.5-7B-Instruct"
else
    CONDITIONS="C0_trueman_full,C3_frozen"
    SEEDS="0,1"
    MODELS="Qwen/Qwen2.5-7B-Instruct"
fi

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
            $PYTHON -m experiments.v2_ambitious.run_v2 --stage stage4 \
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
echo "Pipeline complete. Checking status..."
echo "============================================"
$PYTHON -m experiments.v2_ambitious.run_v2 --status

echo ""
echo "Results summary:"
if [[ -f "experiments/v2_ambitious/results/v2_summary.json" ]]; then
    $PYTHON -c "
import json
s = json.load(open('experiments/v2_ambitious/results/v2_summary.json'))
print('Hypothesis verdicts:')
for h, v in s.get('hypothesis_verdicts', {}).items():
    print(f'  {h}: {v}')
"
else
    echo "  (No summary file yet — run stage6 to generate)"
fi
