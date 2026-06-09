#!/usr/bin/env bash
# One-shot clean rerun of the full v2 pipeline (stage0 -> stage6).
#
# Run this AFTER pulling the fixes for: stage-1 per-step slowness + capture
# misalignment, anxiety-signal variance, LoRA-expert attachment, and the
# off->False / empty-feature / FDR selection fixes.
#
# Why a full --force rerun:
#   * All previous stage-1 data was produced by the old (slow, misaligned,
#     saturated-anxiety, non-plastic) code, so it is not comparable.
#   * The stimulus stream is now longer (14d x 24h = 336 steps), so the old
#     84/168-step stream file must be rebuilt or stage-1 aborts ("stream too
#     short").
#
# Long run (~16-20h). Launch it detached, e.g.:
#     nohup bash experiments/v2_ambitious/rerun_v2.sh > rerun.out 2>&1 &
#     tail -f experiments/v2_ambitious/results/run_v2.log
#
set -u

# cd to repo root (this script lives in experiments/v2_ambitious/)
cd "$(dirname "$0")/../.." || exit 1

PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="python"
echo "[rerun] repo: $(pwd)"
echo "[rerun] python: $PY"

echo "[rerun] removing stale DERIVED results (code, configs and stage0 probes kept)..."
rm -rf experiments/v2_ambitious/results/longhorizon
rm -rf experiments/v2_ambitious/results/mechanistic
rm -rf experiments/v2_ambitious/results/indicators
rm -rf experiments/v2_ambitious/results/subprocess_logs
rm -f  experiments/v2_ambitious/results/analysis_*.json
rm -f  experiments/v2_ambitious/results/fep_*.json
rm -f  experiments/v2_ambitious/results/v2_summary.json
# Old stream is shorter than the new 336-step config -> force a rebuild.
rm -f  experiments/v2_ambitious/data/stimulus_stream.jsonl
# Old LoRA experts were trained under the broken (non-attaching) path -> clear
# so plasticity starts from a clean pool.
rm -rf adapters
# Stale run_v2.log accumulates across runs (append mode); archive it so the new
# run is easy to read from the top.
[ -f experiments/v2_ambitious/results/run_v2.log ] && \
  mv experiments/v2_ambitious/results/run_v2.log \
     "experiments/v2_ambitious/results/run_v2.log.$(date +%Y%m%d_%H%M%S).bak"

echo "[rerun] launching full pipeline: stage0..stage6, --force, seeds 0,1"
"$PY" -m experiments.v2_ambitious.run_v2 --stage all --force --seeds 0,1
rc=$?

echo "[rerun] run_v2 exit code: $rc"
echo "[rerun] log:     experiments/v2_ambitious/results/run_v2.log"
echo "[rerun] summary: experiments/v2_ambitious/results/v2_summary.json"
exit $rc
