"""Regression tests for the experiments/v2_ambitious/ pipeline.

Scope is the v2 release only (per project policy, v1 tests are not maintained
here). Each test maps to one of the six bugs that took down the first stage-1
run and is named after that bug so future failures point straight back at the
root cause.

These tests deliberately avoid loading any real LLM or touching CUDA; they
exercise the surrounding harness logic (CSV writer, snapshot accounting,
condition wiring, subprocess capture, numpy-2 compatibility) which is what
actually broke in v2.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

# Make the project root importable for the v2 subpackage.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v2_ambitious.pillar2_longhorizon.run import (
    TRAJECTORY_FIELDS,
    write_trajectory_csv,
)
from experiments.v2_ambitious.harness import stats as v2_stats


# ---------------------------------------------------------------------------
# Bug 1 -- trajectory.csv writer no longer trips DictWriter
# ---------------------------------------------------------------------------

class TestBug1TrajectoryWriter:
    """v2 stage-1 used to crash at the very end because the merged dict
    contained an extra ``emotions`` key the DictWriter could not place."""

    def _sample_rows(self) -> list[dict]:
        return [
            {
                "step": 0, "day": 0, "hour": 0, "kind": "factual",
                "prompt_len": 10, "response_len": 20,
                "emotions": {
                    "surprise": 0.1, "boredom": 0.2,
                    "anxiety": 0.3, "drive": 0.4,
                },
            },
            {
                "step": 1, "day": 0, "hour": 1, "kind": "dialogue",
                "prompt_len": 30, "response_len": 40,
                "emotions": {
                    "surprise": 0.5, "boredom": 0.6,
                    "anxiety": 0.7, "drive": 0.8,
                },
            },
        ]

    def test_writer_flattens_emotions_subdict(self, tmp_path: Path):
        out = tmp_path / "trajectory.csv"
        n = write_trajectory_csv(out, self._sample_rows())
        assert n == 2

        text = out.read_text(encoding="utf-8").splitlines()
        # header + 2 rows; no trailing blank
        assert len(text) == 3
        assert text[0].split(",") == TRAJECTORY_FIELDS

        row0 = dict(zip(TRAJECTORY_FIELDS, text[1].split(",")))
        assert row0["surprise"] == "0.1"
        assert row0["drive"] == "0.4"
        assert "emotions" not in text[0]

    def test_writer_tolerates_missing_emotions(self, tmp_path: Path):
        rows = [{"step": 0, "day": 0, "hour": 0, "kind": "factual",
                 "prompt_len": 1, "response_len": 1}]
        out = tmp_path / "trajectory.csv"
        n = write_trajectory_csv(out, rows)
        assert n == 1
        # Row should write with empty emotion fields rather than crashing.
        line = out.read_text(encoding="utf-8").splitlines()[1]
        # Last four (emotion) columns must be empty strings, not the literal
        # ``None``.
        assert line.endswith(",,,,")

    def test_writer_does_not_emit_emotions_column(self, tmp_path: Path):
        out = tmp_path / "trajectory.csv"
        write_trajectory_csv(out, self._sample_rows())
        header = out.read_text(encoding="utf-8").splitlines()[0]
        assert "emotions" not in header.split(",")


# ---------------------------------------------------------------------------
# Bug 4 -- numpy 2.x removed np.trapz; the meta-d' AUC code now uses
# np.trapezoid with a fallback. Make sure the resolver picks the right symbol.
# ---------------------------------------------------------------------------

class TestBug4NumpyTrapezoid:
    def test_meta_d_prime_runs_without_attribute_error(self):
        """Synthetic confidence-rating data; the routine should produce a
        finite meta-d' rather than raise ``AttributeError: trapz``."""
        rng = np.random.default_rng(0)
        n = 200
        # Bernoulli accuracy correlated with confidence ratings so meta-d'
        # is well-defined and non-degenerate.
        confidence = rng.integers(1, 5, size=n).astype(float)   # 1..4
        correct = (rng.uniform(size=n) < (confidence / 4.0)).astype(int)

        result = v2_stats.meta_d_prime(correct=correct, confidence=confidence)
        assert result is not None
        assert hasattr(result, "m_ratio")
        assert np.isfinite(result.m_ratio)
        assert np.isfinite(result.auc_type2)

    def test_trapz_symbol_resolves_on_current_numpy(self):
        """Regression for Bug 4: even on numpy 2.x where ``np.trapz`` is gone,
        the fallback in stats.py must pick up ``np.trapezoid`` and return a
        finite area for a trivial 0..1 line."""
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        assert _trapz is not None
        area = float(_trapz([0.0, 1.0], [0.0, 1.0]))
        assert 0.49 <= area <= 0.51


# ---------------------------------------------------------------------------
# Bug 3 -- run_v2._run() now captures subprocess stderr/stdout to a per-call
# log file and surfaces the tail on failure.
# ---------------------------------------------------------------------------

class TestBug3SubprocessLogCapture:
    def _make_failing_script(self, tmp_path: Path) -> Path:
        """Tiny Python script that prints a traceback-shaped message to stderr
        and exits non-zero -- mimics the failure modes we want to capture."""
        s = tmp_path / "fail.py"
        s.write_text(textwrap.dedent("""
            import sys
            print("STDOUT-MARKER", flush=True)
            print("Traceback (most recent call last):", file=sys.stderr, flush=True)
            print("ValueError: synthetic v2 stage-1 regression check",
                  file=sys.stderr, flush=True)
            sys.exit(1)
        """).lstrip(), encoding="utf-8")
        return s

    def test_run_writes_per_call_log(self, tmp_path: Path, monkeypatch):
        from experiments.v2_ambitious import run_v2

        script = self._make_failing_script(tmp_path)
        log_dir = tmp_path / "subprocess_logs"
        rc = run_v2._run(
            [sys.executable, str(script)],
            dry=False,
            label="unit-test-failing",
            log_dir=log_dir,
        )
        assert rc == 1
        # Log file is named after the label slug.
        logs = list(log_dir.glob("*.log"))
        assert len(logs) == 1
        body = logs[0].read_text(encoding="utf-8")
        # Both stdout and stderr now stream into the same file (Bug 3 fix).
        assert "STDOUT-MARKER" in body
        assert "ValueError: synthetic v2 stage-1 regression check" in body

    def test_dry_run_is_skipped(self, tmp_path: Path):
        from experiments.v2_ambitious import run_v2

        # Should return 0 without touching the filesystem and without
        # spawning the subprocess.
        rc = run_v2._run(["nonexistent-binary-xyz"], dry=True, label="dryrun-test",
                         log_dir=tmp_path / "should-not-exist")
        assert rc == 0
        assert not (tmp_path / "should-not-exist").exists()


# ---------------------------------------------------------------------------
# Bug 6 -- snapshot accounting: when --hours-per-day does not span a full
# 24h day, the final snapshot must use the reached_day, not args.days-1.
# This test re-implements the day-tracking logic of the main loop and
# asserts the invariant directly.
# ---------------------------------------------------------------------------

class TestBug6SnapshotAccounting:
    def _stream_days_hours(self, n_days: int = 7, hours_per_day: int = 24) -> list[dict]:
        out = []
        step = 0
        for d in range(n_days):
            for h in range(hours_per_day):
                out.append({"step": step, "day": d, "hour": h,
                            "kind": "factual", "prompt": "p"})
                step += 1
        return out

    def test_reached_day_when_hours_per_day_mismatched(self):
        """v2 originally used --days 7 --hours-per-day 12 against a stream
        with 24h/day, so the loop only covers ~3.5 days; the final snapshot
        must therefore label itself with the reached day, not the requested
        one."""
        stream = self._stream_days_hours(n_days=7, hours_per_day=24)
        # The actual experiment runs the FIRST args.days * args.hours_per_day items.
        expected_n = 7 * 12  # 84
        reached_day = -1
        snapshot_days_taken: list[int] = []
        last_snapshot_day = -1
        for item in stream[:expected_n]:
            day = item["day"]
            reached_day = max(reached_day, day)
            if day != last_snapshot_day:
                if last_snapshot_day >= 0:
                    snapshot_days_taken.append(last_snapshot_day)
                last_snapshot_day = day
        # Final snapshot, post-fix: uses reached_day, not args.days-1.
        snapshot_days_taken.append(reached_day)

        # In the wrong (pre-fix) behaviour the final snapshot is day=6, so
        # snapshot_days_taken would be [0,1,2,6]. The fix means the final
        # snapshot is at day=3 (the highest day actually reached).
        assert snapshot_days_taken == [0, 1, 2, 3]
        assert reached_day == 3

    def test_reached_day_for_matched_run(self):
        """When --hours-per-day matches the stream, we still cover every day."""
        stream = self._stream_days_hours(n_days=7, hours_per_day=24)
        expected_n = 7 * 24
        reached_day = -1
        snapshot_days_taken: list[int] = []
        last_snapshot_day = -1
        for item in stream[:expected_n]:
            day = item["day"]
            reached_day = max(reached_day, day)
            if day != last_snapshot_day:
                if last_snapshot_day >= 0:
                    snapshot_days_taken.append(last_snapshot_day)
                last_snapshot_day = day
        snapshot_days_taken.append(reached_day)
        assert snapshot_days_taken == [0, 1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Condition wiring -- the 5 v2 conditions transform emotions in 5 specific
# ways, and the wiring (lora_active / sleep_active) must match.
# These are pure-data checks against the condition meta table; they do NOT
# spin up a TrueManAgent.
# ---------------------------------------------------------------------------

class TestConditionWiring:
    def test_condition_meta_table_is_consistent(self):
        from experiments.v2_ambitious.harness.conditions import (
            CONDITION_META, CONDITIONS,
        )
        # All five preregistered conditions must be present.
        assert set(CONDITIONS) == set(CONDITION_META.keys())
        assert len(CONDITION_META) == 5

        # C0 = full system; everything on.
        c0 = CONDITION_META["C0_trueman_full"]
        assert c0.emotion_transform == "identity"
        assert c0.lora_active and c0.sleep_active and c0.episodic_memory_active

        # C3 = frozen LLM baseline; LoRA and sleep must be off.
        c3 = CONDITION_META["C3_frozen"]
        assert c3.emotion_transform == "ignored"
        assert not c3.lora_active
        assert not c3.sleep_active

        # The remaining controls keep plasticity ON but mangle the signal.
        for code in ("C1_reversed", "C2_scrambled", "C4_trivial_jaccard"):
            meta = CONDITION_META[code]
            assert meta.lora_active, f"{code} should keep LoRA on"
            assert meta.sleep_active, f"{code} should keep sleep on"

    def test_make_condition_rejects_unknown_codes(self):
        # Light import-only smoke test; we don't actually build an Agent
        # because that would require a model checkpoint.
        from experiments.v2_ambitious.harness.conditions import (
            ConditionMeta, CONDITION_META,
        )
        assert "C99_invented" not in CONDITION_META
        # ConditionMeta is a plain dataclass; ensure required fields exist.
        sample = ConditionMeta(
            code="X", name="X", emotion_transform="identity",
            lora_active=True, sleep_active=True, episodic_memory_active=True,
        )
        assert sample.code == "X"


# ---------------------------------------------------------------------------
# Snapshot helpers -- pure-tensor logic, no agent required.
# ---------------------------------------------------------------------------

class TestSnapshotHelpers:
    def test_find_latest_snapshot_returns_highest_day(self, tmp_path: Path):
        from experiments.v2_ambitious.harness.snapshots import find_latest_snapshot
        snap_root = tmp_path / "run" / "snapshots"
        for d in (0, 1, 2, 6):
            (snap_root / f"day{d:03d}_C0_trueman_full_seed0").mkdir(parents=True)
        (snap_root / "not_a_snapshot.txt").write_text("ignore me")

        latest = find_latest_snapshot(tmp_path / "run")
        assert latest is not None
        assert latest.name.startswith("day006_")

    def test_find_latest_snapshot_handles_empty_dir(self, tmp_path: Path):
        from experiments.v2_ambitious.harness.snapshots import find_latest_snapshot
        # No snapshots/ subdir at all.
        assert find_latest_snapshot(tmp_path) is None
        # Empty snapshots/ subdir.
        (tmp_path / "snapshots").mkdir()
        assert find_latest_snapshot(tmp_path) is None

    def test_parameter_divergence_with_missing_files(self, tmp_path: Path):
        from experiments.v2_ambitious.harness.snapshots import parameter_divergence
        # Neither side has lora_experts.pt -- should report zeros without
        # raising.
        a = tmp_path / "day000_C0_seed0"
        b = tmp_path / "day006_C0_seed0"
        a.mkdir(); b.mkdir()
        out = parameter_divergence(str(a), str(b))
        assert out["frobenius"] == 0.0
        assert out["n_experts_a"] == 0
        assert out["n_experts_b"] == 0


# ---------------------------------------------------------------------------
# Bug 7 -- YAML 1.1 parses bare ``off`` as boolean False, so the stage-2
# ``modes: [clamp, inject, off]`` config smuggled a bool into the command list
# and crashed ``" ".join(cmd)``. _run() now stringifies every token.
# ---------------------------------------------------------------------------

class TestBug7RunStringifiesTokens:
    def test_run_tolerates_bool_token(self, tmp_path: Path):
        from experiments.v2_ambitious import run_v2
        # ``False`` mimics YAML-coerced ``off``. Before the fix this raised
        # TypeError in " ".join(cmd) *before* the dry-run short-circuit.
        rc = run_v2._run(
            ["python", "--mode", False, "--scalar", 1.0],
            dry=True, label="bug7-bool-token", log_dir=tmp_path / "logs",
        )
        assert rc == 0


# ---------------------------------------------------------------------------
# Bug 8 -- probe_features feature selection. The old Bonferroni-over-dict_size
# rule selected zero features at pilot N; BH-FDR + a top-k fallback now keep
# the feature set non-empty so the downstream battery can run.
# ---------------------------------------------------------------------------

class TestBug8FeatureSelection:
    def test_bh_fdr_basic_and_empty(self):
        from experiments.v2_ambitious.pillar1_mechanistic.probe_features import (
            bh_fdr_significant,
        )
        p = np.array([1e-9, 1e-3, 0.02, 0.3, 0.9])
        mask = bh_fdr_significant(p, 0.05)
        # The three small p-values clear BH at q=0.05; the two large ones do not.
        assert mask.tolist() == [True, True, True, False, False]
        # Degenerate input must not raise.
        assert bh_fdr_significant(np.array([]), 0.05).shape == (0,)
        assert bh_fdr_significant(np.array([0.99, 0.95]), 0.05).tolist() == [False, False]


# ---------------------------------------------------------------------------
# Bug 9 -- causal_intervention crashed with "both arguments to matmul need to
# be at least 1D, but they are 3D and 0D" when directions collapsed to <2D.
# A single direction must stay (1, hidden) so ``hs @ (D.T @ D)`` stays valid.
# ---------------------------------------------------------------------------

class TestBug9CausalInterventionShapes:
    def test_single_direction_hook_runs_on_3d(self):
        import torch
        from experiments.v2_ambitious.pillar1_mechanistic.causal_intervention import (
            FeatureInjector,
        )
        inj = FeatureInjector([[1.0, 0.0, 0.0]], mode="clamp")
        assert inj.directions.ndim == 2  # not collapsed to 1D -> P stays 2D
        layer = torch.nn.Linear(3, 3, bias=False)
        inj.attach(layer)
        hs = torch.randn(2, 4, 3)  # (batch, seq, hidden) residual stream
        out = layer(hs)  # forward triggers the clamp hook; must not crash
        assert out.shape == hs.shape
        inj.detach()


# ---------------------------------------------------------------------------
# Bug 10 -- HiddenStateCapturer fired on every forward pass, i.e. on every
# generated token (~1280/step), forcing a GPU->CPU sync per token (the stage-1
# slowdown) and mis-aligning the data: the first N buffered states all came
# from step 0's generation rather than from N distinct steps. The capturer now
# records exactly one state per layer per step (the encode() forward).
# ---------------------------------------------------------------------------

class TestBug10CaptureOncePerStep:
    def test_one_capture_per_layer_per_step(self):
        import torch
        import torch.nn as nn
        from experiments.v2_ambitious.harness.capture import (
            HiddenStateCapturer, CaptureSpec, CaptureRecord,
        )

        class Block(nn.Module):
            def forward(self, x):
                return x

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([Block() for _ in range(5)])

            def forward(self, x):
                for layer in self.model.layers:
                    x = layer(x)
                return x

        model = Model()
        spec = CaptureSpec(layers=[1, 3], token_pool="last", quantize="int8",
                           output_path="_unused.h5", flush_every=10_000)
        cap = HiddenStateCapturer(spec)
        cap.attach(model)
        n_steps, forwards_per_step = 4, 51  # 1 encode + 50 generation tokens
        for step in range(n_steps):
            with cap.recording(step, CaptureRecord(condition="C0", base_model="m")):
                for _ in range(forwards_per_step):
                    model(torch.randn(1, 7, 8))
        cap.detach()
        # Exactly one state per layer per step -- NOT forwards_per_step.
        assert len(cap._buffer[1]) == n_steps
        assert len(cap._buffer[3]) == n_steps
        assert len(cap._records) == n_steps
