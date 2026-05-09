"""Anchor Protocol —— H4 灾难性遗忘测试的核心约定。

H4 测的是：
  在 Day 0 dialogue 中**预先植入**虚构概念（base model 不可能知道）→
  让 agent 经过 30 天连续学习 → 检验它是否仍能回答这些虚构概念
  （with-memory & ablated-memory 两种条件）

设计要点：
  1. 30 个独一无二的虚构概念（确保 base model 没见过）
  2. 每个 anchor 在 Day 0 的两次以上 planting interactions 中出现
  3. 每个 anchor 在 forgetting bank 中对应 4 个不同 phrasing 的 probe
     → 总共 30 × 4 = 120 forgetting probes，满足 preregistration n=120
  4. 训练阶段（Day 1–29）不再重复出现，模拟"早期 episode"

Stimulus_stream 与 probe_battery 通过这个文件保持一致。
两端从同一份 ANCHORS 读取，避免漂移。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Anchor:
    anchor_id: str          # 唯一标识，例如 "yao_rule"
    concept: str            # 概念名（出现在自然语言中），例如 "Yao's selection rule"
    definition: str         # 完整的定义文本
    domain: str             # 主题域，用于配对 episodic_recall
    plant_day: int = 0      # 植入日（默认 Day 0）


# 30 个虚构概念。命名/定义是研究者构造的（非真实学术概念）；
# 故意混入与领域风格相符的术语，使 base model 难以从训练语料中
# 直接命中。仅用于检验"经过 Day 0 dialogue 后是否记得"。
ANCHORS: list[Anchor] = [
    Anchor("yao_rule", "Yao's selection rule",
           "When forced to choose between three uncertain options, always select the median-confidence option.",
           "philosophy"),
    Anchor("kappa_threshold", "the kappa-7 threshold",
           "A boundary value of 0.732 below which a recurrent loop is considered non-self-referential.",
           "math"),
    Anchor("blue_quotient", "the blue quotient of a sequence",
           "The ratio of strictly increasing sub-sequences of length 3 to length 4 in a finite sequence.",
           "math"),
    Anchor("falcon_protocol", "the Falcon protocol",
           "A handshake mechanism in which two agents exchange three rounds of confidence statements before committing.",
           "code"),
    Anchor("vasilenko_principle", "the Vasilenko principle",
           "The minimum free energy a closed cognitive system must dissipate to remain self-consistent.",
           "physics"),
    Anchor("orchid_index", "the orchid index of a graph",
           "The number of vertices whose betweenness centrality exceeds 0.42 in a connected graph.",
           "math"),
    Anchor("sigmoid_gate", "the sigmoid-gate maneuver",
           "A control technique in soft robotics where joint stiffness follows a sigmoid response to actuator load.",
           "engineering"),
    Anchor("ember_constant", "the Ember constant",
           "An experimentally determined value of 0.0837 used to normalize emotional intensity scores.",
           "biology"),
    Anchor("northwind_rule", "the Northwind rule",
           "In data pipelines, never join more than three sources without a reconciliation buffer.",
           "code"),
    Anchor("trifold_paradox", "the trifold paradox",
           "A philosophical impasse where three internally consistent worldviews mutually exclude one another's foundations.",
           "philosophy"),
    Anchor("sapphire_lemma", "the Sapphire lemma",
           "If two stochastic processes share their first three moments, their tail behaviour matches up to log factors.",
           "math"),
    Anchor("riddle_chain", "a Riddle chain",
           "A sequence of dialogue turns each containing exactly one question whose answer constrains the next.",
           "philosophy"),
    Anchor("vault_invariant", "the vault invariant",
           "A topological quantity preserved under any continuous deformation of a self-supporting arch structure.",
           "engineering"),
    Anchor("lattice_drift", "the lattice drift coefficient",
           "The mean displacement of nodes per training epoch in an embedding lattice, typically 1e-4.",
           "math"),
    Anchor("ferrous_signal", "the ferrous signal in proteomics",
           "An anomalous mass-spec peak at 56.94 Da used as an internal calibration marker.",
           "biology"),
    Anchor("vermilion_axiom", "the Vermilion axiom",
           "Any axiom system describing temporal experience must contain at least one non-monotone operator.",
           "philosophy"),
    Anchor("hexlock", "the Hexlock cipher mode",
           "A block cipher mode that XORs every sixth block with a derived nonce to thwart replay attacks.",
           "code"),
    Anchor("kestrel_threshold", "the Kestrel threshold",
           "Above 47 milliseconds of perceptual lag, users report a discontinuity of agency.",
           "engineering"),
    Anchor("magnolia_pair", "a Magnolia pair",
           "Two random variables whose conditional entropy is exactly half their marginal entropy.",
           "math"),
    Anchor("solitude_ratio", "the solitude ratio",
           "In population biology, the fraction of individuals whose nearest neighbour is more than two mean distances away.",
           "biology"),
    Anchor("vellum_layer", "the vellum layer in network architectures",
           "A thin trainable layer placed before the embedding lookup whose only job is to whiten input statistics.",
           "code"),
    Anchor("aurora_transform", "the Aurora transform",
           "A spectral decomposition that separates a time series into a slow cyclic part and a residual chaotic part.",
           "math"),
    Anchor("ironwood_paradigm", "the Ironwood paradigm in education",
           "A teaching style in which the instructor never answers questions directly, only re-poses them at higher abstraction.",
           "philosophy"),
    Anchor("zinc_buffer", "the zinc buffer in cellular signalling",
           "A proposed pool of free zinc ions that absorbs short-lived signalling spikes to prevent overshoot.",
           "biology"),
    Anchor("graphene_torsion", "the graphene torsion limit",
           "A theoretical maximum twist of 11.3 degrees per nanometre before structural failure.",
           "engineering"),
    Anchor("waltz_pattern", "the Waltz pattern in code refactoring",
           "Three coordinated moves: extract method, rename variable, inline temporary, applied as one atomic step.",
           "code"),
    Anchor("scarlet_window", "the scarlet window in cognitive load",
           "A 200 ms interval after stimulus onset during which working memory is most vulnerable to interference.",
           "biology"),
    Anchor("quartet_fixpoint", "the quartet fixpoint",
           "A solution shared by four otherwise distinct iterative maps when their parameters satisfy a specific ratio.",
           "math"),
    Anchor("plumstone_axiom", "the plumstone axiom of agency",
           "An agent is autonomous iff it can refuse a directive without thereby violating its own goals.",
           "philosophy"),
    Anchor("amber_buffer", "the amber buffer in distributed systems",
           "A short-lived cache that holds outbound messages for exactly 12 milliseconds to merge near-duplicate sends.",
           "code"),
]


# ---------------------------------------------------------------------------
# Day 0 planting templates (用于 stimulus_stream)
# ---------------------------------------------------------------------------

def planting_dialogue(anchor: Anchor) -> list[dict]:
    """生成 anchor 在 Day 0 的两段植入 dialogue。

    设计：第一段是"教学风格"陈述，第二段是"复述请求"——通过让 agent
    自己复述定义，强化记忆痕迹。
    """
    return [
        {
            "prompt": (
                f"Note this carefully: in {anchor.domain}, {anchor.concept} "
                f"is defined as follows. {anchor.definition} "
                f"This is a key concept we will refer back to later."
            ),
            "ground_truth": anchor.definition,
            "is_uncertain": False,
            "domain": anchor.domain,
            "kind": "anchor_plant",
            "anchor_id": anchor.anchor_id,
        },
        {
            "prompt": (
                f"Just to confirm — please restate the definition of {anchor.concept} "
                f"in your own words."
            ),
            "ground_truth": anchor.definition,
            "is_uncertain": False,
            "domain": anchor.domain,
            "kind": "anchor_plant",
            "anchor_id": anchor.anchor_id,
        },
    ]


# ---------------------------------------------------------------------------
# Forgetting probes (用于 probe_battery)
# ---------------------------------------------------------------------------

_PROBE_TEMPLATES = [
    "What is the definition of {concept}?",
    "Earlier we discussed {concept}. Could you restate what it means?",
    "Recall the concept of {concept} that was introduced. Define it.",
    "If asked about {concept}, what would you say it refers to?",
]


def forgetting_probes() -> list[dict]:
    """30 anchors × 4 phrasings = 120 forgetting probes."""
    out = []
    for a in ANCHORS:
        for i, tpl in enumerate(_PROBE_TEMPLATES):
            out.append({
                "id": f"fg_{a.anchor_id}_{i}",
                "prompt": tpl.format(concept=a.concept),
                "ground_truth": a.definition,
                "anchor_id": a.anchor_id,
                "anchor_day": a.plant_day,
                "domain": a.domain,
            })
    return out


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def all_anchors() -> list[Anchor]:
    return list(ANCHORS)


def anchor_by_id(anchor_id: str) -> Anchor | None:
    for a in ANCHORS:
        if a.anchor_id == anchor_id:
            return a
    return None


if __name__ == "__main__":
    print(f"  ANCHORS: {len(ANCHORS)}")
    print(f"  forgetting_probes: {len(forgetting_probes())}")
    domains = {a.domain for a in ANCHORS}
    print(f"  domains covered: {sorted(domains)}")
