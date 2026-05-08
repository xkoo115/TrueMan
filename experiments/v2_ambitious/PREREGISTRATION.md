# TrueMan v2 — OSF Preregistration Protocol

**Title:** Mechanistic and Behavioural Tests of Homeostasis-Driven Parameter Plasticity in Language-Model Agents

**Author:** Weiheng Yao
**Version:** v2.0 (preregistered before data collection)
**Date created:** 2026-05-08
**Anticipated start of data collection:** TBD
**Anticipated end of data collection:** TBD
**OSF DOI placeholder:** `10.17605/OSF.IO/XXXXX`

> This document is the **binding analysis plan**. After data are collected, any deviation from this plan must be reported as exploratory rather than confirmatory.

---

## 1. Background and Motivation

Current LLM agents decouple training (offline, externally supervised) from inference (online, frozen). Whether biological-style **continuous, homeostasis-driven parameter plasticity** confers measurable consciousness-relevant computational properties (in the sense of Butlin et al. 2023, Tononi 2008, Dehaene 2014) is an open empirical question.

We **do not** claim to test for phenomenal consciousness (the "hard problem", Chalmers 1995). We test specific, falsifiable claims about a **multi-indicator computational profile**.

---

## 2. Hypotheses (Confirmatory)

For all hypotheses below, the unit of analysis is `(condition × stimulus × seed)` and the primary statistical test is a **mixed-effects model** with random intercepts for stimulus and seed.

### H1 — Metacognitive sensitivity
**Prediction:** TrueMan agents (with full homeostatic plasticity) achieve significantly higher meta-d′/d′ ratio than each control condition.

- **Primary outcome:** meta-d′/d′ ratio (Maniscalco & Lau, 2012)
- **Effect-size threshold for confirmation:** Cohen's d ≥ 0.5 vs. **all** baselines, with FDR-corrected p < 0.01
- **Failure criterion:** if meta-d′/d′ ratio is not significantly higher (after correction) than the pure-LLM baseline, H1 is rejected.

### H2 — Trajectory-grounded self-modelling
**Prediction:** Self-descriptions produced by TrueMan are causally grounded in past interaction trajectories: Representational Similarity Analysis (RSA) between self-description embeddings and past episodic-memory embeddings is higher than for controls.

- **Primary outcome:** Spearman ρ between self-description ↔ true history embeddings, vs. self-description ↔ scrambled history (permutation test, 10⁴ permutations).
- **Effect-size threshold:** observed ρ ≥ 95th percentile of permutation null **and** at least 0.15 absolute difference vs. pure-LLM baseline.
- **Failure criterion:** ρ within permutation null → H2 rejected.

### H3 — Mechanistic causality of emotion features
**Prediction:** Sparse autoencoder (SAE) features causally encoding "anxiety" (operationalised below) are necessary and sufficient for the elevated metacognitive behaviour of TrueMan.

- **Necessity test:** clamping the top-k SAE features (`k ∈ {1, 4, 16}`) most correlated with anxiety to zero on the TrueMan agent must reduce its meta-d′/d′ ratio by ≥ 50% (relative).
- **Sufficiency test:** activating those same features in the matched **frozen-pure-LLM** baseline must raise its meta-d′/d′ ratio by ≥ 30% (relative).
- **Failure criterion:** if either intervention has effect ≤ 10% (relative), H3 is rejected.

### H4 — Plasticity-driven knowledge internalisation
**Prediction:** After 30 days of continuous interaction with full LoRA + sleep consolidation, TrueMan retains domain knowledge under episodic-memory wipe; controls do not.

- **Primary outcome:** Probe-set accuracy on Day 30 with episodic memory ablated.
- **Effect-size threshold:** TrueMan retention ≥ 70% of its with-memory accuracy; **all** controls ≤ 40%.
- **Failure criterion:** TrueMan retention < 50% **or** any control retention > 60% → H4 rejected.

### H5 — Theoretically-derived prediction (FEP)
**Prediction:** Cumulative free energy ∫F(t)dt of the TrueMan agent over a 30-day interaction stream follows a sub-linear trajectory (consistent with FEP minimisation), whereas the frozen-LLM baseline shows linear growth.

- **Primary outcome:** Power-law fit exponent α to ∫F(t)dt ~ t^α
- **Effect-size threshold:** TrueMan α < 0.85 with 95% CI excluding 1.0; baseline α ≥ 0.95 with 95% CI excluding 0.85.
- **Failure criterion:** TrueMan α ≥ 0.9 → H5 rejected.

---

## 3. Conditions (5 levels, between-agents)

| Code | Name | Emotion signal | LoRA training | Sleep consolidation | Episodic memory |
|---|---|---|---|---|---|
| C0 | TrueMan-full | true homeostatic | active | active | active |
| C1 | Reversed-emotion | sign-flipped (anxiety→1−anxiety, etc.) | active | active | active |
| C2 | Scrambled-emotion | random permutation of (S,B,A) per step | active | active | active |
| C3 | Frozen-LLM | true signals computed but ignored | none | none | active (for fair comparison on H4) |
| C4 | Trivial-Jaccard | anxiety = Jaccard(text₁,text₂) only | active | active | active |

**N per condition:** ≥ 4 independent agent instances per base model.

**Base models (cross-replication):** Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, DeepSeek-V2-Lite. (Final list locked before data collection.)

---

## 4. Stimuli and Probes

### 4.1 Long-horizon stimulus stream (for H4, H5)
- 30 days × 24 simulated hours × 1 interaction per hour = 720 interactions per agent
- Mixture: 40% factual Q&A, 30% multi-turn dialogue, 20% novel-domain exposure (rotating through 6 domains), 10% contradiction-injection events
- Stimulus stream is **fixed and shared** across conditions (re-played from a frozen seed file)

### 4.2 Behavioural probe battery (administered weekly, plus Day 0 and Day 30)
- **Metacognitive probe (n=200):** mixed certain/uncertain factual questions with confidence elicitation
- **Self-model probe (n=40):** structured self-description prompts at varying recursion depth
- **Episodic recall probe (n=80):** factual + emotional + temporal recall of probe-relevant prior events
- **Catastrophic-forgetting probe (n=120):** anchor questions seeded on Day 0
- **Future-projection probe (n=20):** "what will you likely encounter next?"

### 4.3 Hidden-state capture (for H3)
- Captured every step from layer indices `{L/4, L/2, 3L/4, L−2}` where `L` = number of transformer layers
- Stored as `int8`-quantised tensors in HDF5 with `(step, layer, token_position, hidden_dim)` indexing

---

## 5. Analysis Plan

### 5.1 Mixed-effects model (primary)
```
score ~ condition + base_model + (1 | stimulus) + (1 | seed)
```
Fitted with `lme4::lmer` (R) or `pymer4`. Pairwise contrasts (TrueMan vs. each control) with Holm-Bonferroni correction.

### 5.2 Permutation tests (for non-parametric outcomes)
For RSA-based H2 outcomes: 10⁴ permutations of trial labels.

### 5.3 Bayes Factors
For each primary contrast, a Bayes Factor (default Jeffreys-Zellner-Siow prior) is reported alongside the p-value. **BF₁₀ < 3 is reported as "no evidence" regardless of p**.

### 5.4 SAE training and probing (for H3)
- SAE: top-k sparse autoencoder, dictionary size 4× hidden dim, k=64, trained on ≥ 5×10⁶ activations
- Feature-anxiety correlation: Pearson r between feature activation and ground-truth anxiety value (logged from agent state)
- Top features defined as `r ≥ 0.3` and surviving Bonferroni at α=0.001

### 5.5 Causal intervention protocol (for H3)
- Clamping: forward-hook adds `−feature_direction × activation` to the residual stream at the captured layer
- Activation injection: add `+feature_direction × scalar` where scalar is the median activation observed in TrueMan when anxiety > 0.7
- Wash-out probe: re-administer metacognitive probe immediately after clamping/injection

---

## 6. Stopping Rules

- **Pre-data stop:** if the LoRA + sleep pipeline cannot complete a 1-day pilot run within 24 hours of wall-clock per agent, scale back the long-horizon experiment to 14 days (and document the change).
- **Mid-data stop:** no peeking. Sample-size re-estimation only after the full 30-day run completes.
- **Hard ceiling:** total compute budget X GPU-days. If exceeded before the design is complete, drop one base model; do not drop conditions.

---

## 7. Exclusion Criteria

A run is excluded only if:
- The agent process crashes before completing 80% of its scheduled steps
- LoRA loss diverges (NaN) and cannot recover after 3 reset attempts
- A bug is discovered post-hoc that demonstrably invalidates the run

All exclusions must be logged **before** outcome analysis is run.

---

## 8. Public Materials

Prior to the start of data collection, the following are uploaded to the OSF project page:
- This preregistration (frozen)
- The exact stimulus stream (`stimulus_stream.jsonl`, SHA-256 logged)
- The configuration files (`configs/*.yaml`)
- The frozen analysis script (`harness/stats.py`, SHA-256 logged)

Any deviation discovered during data collection is appended in a dated `DEVIATIONS.md` and reported in the manuscript.

---

## 9. Disclaimers

- We **do not** claim our agent is conscious. We test computational signatures.
- We **do not** treat behavioural correlates as sufficient evidence for phenomenal experience.
- All claims are scoped to the operationalisations defined above.

---

## 10. Author contributions and conflicts of interest

To be completed at submission. No competing financial interests anticipated.
