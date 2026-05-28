"""Analyze the only set of valid numerical observations from the v2 run: Φ^R.

Reads results/trueman_results/v2_summary.json, extracts per-condition phi_R values
across the 10 (condition × seed) runs, computes descriptive statistics and
plastic-vs-frozen contrasts, and produces two figures used by the manuscript.

Outputs go to docs/sn-article-v2/figures/ and docs/sn-article-v2/analysis/.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / "results" / "trueman_results" / "v2_summary.json"
FIG_DIR = ROOT / "docs" / "sn-article-v2" / "figures"
OUT_DIR = ROOT / "docs" / "sn-article-v2" / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    "C0_trueman_full",
    "C1_reversed",
    "C2_scrambled",
    "C3_frozen",
    "C4_trivial_jaccard",
]
LABELS = {
    "C0_trueman_full":  "C0\nTrueMan-full",
    "C1_reversed":      "C1\nReversed",
    "C2_scrambled":     "C2\nScrambled",
    "C3_frozen":        "C3\nFrozen-LLM",
    "C4_trivial_jaccard": "C4\nTrivial-Jaccard",
}
COLOURS = {
    "C0_trueman_full":  "#1f6feb",
    "C1_reversed":      "#8957e5",
    "C2_scrambled":     "#bc8cff",
    "C3_frozen":        "#d1242f",
    "C4_trivial_jaccard": "#3fb950",
}


def load_phi_r() -> dict[str, list[float]]:
    with open(SUMMARY, "r", encoding="utf-8") as f:
        summary = json.load(f)
    per_cond: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    for entry in summary["indicators_summary"]:
        cond = entry["condition"]
        phi = entry["indicators"].get("phi_R", {}).get("phi_r_approx")
        if phi is not None and cond in per_cond:
            per_cond[cond].append(float(phi))
    return per_cond


def permutation_test(a: np.ndarray, b: np.ndarray, n_resamples: int = 10000, rng=None) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)
    observed = float(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    null = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        perm = rng.permutation(pooled)
        null[i] = float(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
    p_two = float(np.mean(np.abs(null) >= abs(observed)))
    return {
        "observed_diff": observed,
        "p_two_sided": p_two,
        "null_ci_95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "n_resamples": n_resamples,
    }


def cohens_d_pooled(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        # Tiny-sample fallback: use simple SD if available, else NaN
        s = math.hypot(float(np.std(a, ddof=0)), float(np.std(b, ddof=0))) or float("nan")
        return float((np.mean(a) - np.mean(b)) / s) if s == s and s > 0 else float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    s_pool = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if s_pool == 0:
        return float("nan")
    return (float(np.mean(a)) - float(np.mean(b))) / s_pool


def bootstrap_ci(x: np.ndarray, n_resamples: int = 10000, rng=None) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    if len(x) < 2:
        m = float(np.mean(x)) if len(x) else float("nan")
        return (m, m)
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(x, size=len(x), replace=True)
        means[i] = float(np.mean(sample))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def figure_phi_r(per_cond: dict[str, list[float]], stats_rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    means = [np.mean(per_cond[c]) if per_cond[c] else 0.0 for c in CONDITIONS]
    sems  = [(np.std(per_cond[c], ddof=1) / math.sqrt(len(per_cond[c]))) if len(per_cond[c]) >= 2 else 0.0 for c in CONDITIONS]
    xs = np.arange(len(CONDITIONS))
    bar_colors = [COLOURS[c] for c in CONDITIONS]
    bars = ax.bar(xs, means, yerr=sems, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.6, capsize=4)
    # individual seeds
    for x, c in zip(xs, CONDITIONS):
        ys = per_cond[c]
        jitter = (np.random.default_rng(int(x)).uniform(-0.08, 0.08, size=len(ys)))
        ax.scatter(np.full(len(ys), x) + jitter, ys, color="black", zorder=3, s=22)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel(r"$\Phi^{R}$ approximation (nats)")
    ax.set_title(r"Information-integration proxy $\Phi^{R}$ across the five preregistered conditions" + "\n" +
                 r"(Qwen2.5-7B-Instruct, $n{=}2$ seeds per condition, day-6 snapshot)", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    # annotate plastic-vs-frozen gap
    c3_mean = float(np.mean(per_cond["C3_frozen"]))
    ax.axhline(c3_mean, color="#d1242f", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(4.4, c3_mean + 0.1, "C3 mean", color="#d1242f", fontsize=8, ha="right")
    fig.tight_layout()
    out = FIG_DIR / "fig_phi_r.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_pipeline_status() -> Path:
    """Diagnostic figure showing which preregistered outcomes returned valid data."""
    rows = [
        ("H1  HOT-1 meta-d'/d'",      "Stage-1 stream crashed; numpy.trapz API removed"),
        ("H1  HOT-2 probe AUC",       "n = 0 (insufficient eligible items)"),
        ("H1  GWT attention entropy", "Empty; no successful capture pairs"),
        ("H1  RPT reentrance",         "Unavailable; no successful pairs"),
        ("H2  Self-model RSA",        "Not run (PENDING)"),
        ("H3  SAE causal clamp / inject", "Not run (PENDING; SAE training skipped)"),
        ("H4  Memory-ablation retention", "Both with-/without-memory accuracy = 0.0"),
        ("H5  Cumulative-surprise alpha",  "No per-step trajectory written"),
        ("--- Phi_R approximation",       "Returned numerical values for 10/10 runs"),
    ]
    statuses = [r[1] for r in rows]
    labels   = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    colors = ["#d1242f"] * (len(rows) - 1) + ["#1f6feb"]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [1] * len(rows), color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    for yi, lab, st in zip(y, labels, statuses):
        ax.text(0.02, yi, f"  {lab}", ha="left", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(0.98, yi, f"{st}  ", ha="right", va="center", color="white", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ("top", "bottom", "left", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Pipeline outcome status (preliminary release)\nred = not analysable, blue = analysed", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / "fig_pipeline_status.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    per_cond = load_phi_r()
    print("Per-condition phi_R values:")
    for c, vs in per_cond.items():
        print(f"  {c}: n={len(vs)}  values={vs}")

    rng = np.random.default_rng(0)
    rows = []
    for c in CONDITIONS:
        arr = np.array(per_cond[c], dtype=float)
        ci = bootstrap_ci(arr, rng=rng) if len(arr) else (float("nan"), float("nan"))
        rows.append({
            "condition": c,
            "n": int(len(arr)),
            "mean": float(np.mean(arr)) if len(arr) else float("nan"),
            "sd":   float(np.std(arr, ddof=1)) if len(arr) >= 2 else float("nan"),
            "ci_lo": float(ci[0]),
            "ci_hi": float(ci[1]),
            "values": [float(v) for v in arr],
        })
    print("\nDescriptive stats:")
    for r in rows:
        print(f"  {r['condition']:<22} n={r['n']}  mean={r['mean']:.3f}  sd={r['sd']:.3f}  CI=[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]")

    # primary contrast: plastic conditions (C0, C1, C2, C4) vs frozen (C3)
    plastic = np.concatenate([np.asarray(per_cond[c], dtype=float) for c in
                              ("C0_trueman_full", "C1_reversed", "C2_scrambled", "C4_trivial_jaccard")])
    frozen  = np.asarray(per_cond["C3_frozen"], dtype=float)
    contrast = permutation_test(plastic, frozen, rng=np.random.default_rng(1))
    contrast["cohens_d"] = cohens_d_pooled(plastic, frozen)
    contrast["plastic_mean"] = float(np.mean(plastic))
    contrast["frozen_mean"]  = float(np.mean(frozen))

    # secondary: C0 vs C3 (the original preregistered C0-vs-C3 dissociation question)
    c0 = np.asarray(per_cond["C0_trueman_full"], dtype=float)
    c0_vs_c3 = permutation_test(c0, frozen, rng=np.random.default_rng(2))
    c0_vs_c3["cohens_d"] = cohens_d_pooled(c0, frozen)

    # C0 vs all other plastic conditions pooled (does TrueMan-full sit above other plastic controls?)
    other_plastic = np.concatenate([np.asarray(per_cond[c], dtype=float) for c in
                                    ("C1_reversed", "C2_scrambled", "C4_trivial_jaccard")])
    c0_vs_others = permutation_test(c0, other_plastic, rng=np.random.default_rng(3))
    c0_vs_others["cohens_d"] = cohens_d_pooled(c0, other_plastic)

    print("\nPlastic vs Frozen permutation test:")
    print(contrast)
    print("\nC0 vs C3 permutation test:")
    print(c0_vs_c3)
    print("\nC0 vs other plastic (C1+C2+C4) permutation test:")
    print(c0_vs_others)

    payload = {
        "per_condition": rows,
        "contrasts": {
            "plastic_vs_frozen": contrast,
            "C0_vs_C3": c0_vs_c3,
            "C0_vs_other_plastic": c0_vs_others,
        },
    }
    out = OUT_DIR / "phi_r_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out}")

    fig1 = figure_phi_r(per_cond, rows)
    fig2 = figure_pipeline_status()
    print(f"Wrote {fig1}, {fig2}")


if __name__ == "__main__":
    main()
