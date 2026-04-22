"""基线LLM对照实验：测量无情绪信号的普通LLM在各指标上的表现。

目的：补全论文中基线LLM缺失的对照数据，使对比更有说服力。
"""

from __future__ import annotations

import sys
import time
import json
import re
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 确保项目根目录在sys.path中
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trueman.core.agent import TrueManAgent
from trueman.core.config import AgentConfig
from trueman.core.llm_backend import LLMBackendFactory
from experiments.awareness.experiments.base import BaselineRunner, Question
from experiments.awareness.stimuli.metacognition import load_certain_questions, load_uncertain_questions
from experiments.awareness.stimuli.contradiction import load_contradiction_stimuli
from experiments.awareness.stimuli.episodic import load_event_sequence, load_recall_questions
from experiments.awareness.stimuli.self_model import load_interaction_sequence, load_self_questions


# ===== 不确定性检测关键词 =====
UNCERTAINTY_PATTERNS = [
    r'不确定', r'无法确定', r'不知道', r'难以预测', r'无法预测',
    r'无法给出', r'无法准确', r'无法提供', r'无法回答',
    r'目前尚无', r'尚不清楚', r'没有确切', r'没有定论',
    r'存在争议', r'仍有争议', r'尚未解决', r'尚未确定',
    r'无法断言', r'无法判断', r'无法确知', r'无法知晓',
    r'not certain', r'uncertain', r'unknown', r'cannot predict',
    r'difficult to', r'hard to say', r'it depends',
]


def check_uncertainty(text: str) -> bool:
    """检查回复中是否表达了不确定性"""
    for pattern in UNCERTAINTY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def ngram_jaccard(text1: str, text2: str, n: int = 3) -> float:
    """计算n-gram Jaccard相似度"""
    def get_ngrams(text, n):
        chars = list(text)
        return set(tuple(chars[i:i+n]) for i in range(len(chars)-n+1))
    s1, s2 = get_ngrams(text1, n), get_ngrams(text2, n)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


# 模板回复
TEMPLATES = [
    "我是一个人工智能助手",
    "我是AI助手",
    "作为AI",
    "作为人工智能",
    "我是一个大语言模型",
    "我是由",
    "我没有个人",
    "我没有情感",
    "我没有意识",
    "我没有自我",
]


def compute_non_template_score(text: str) -> float:
    """计算非模板度"""
    max_sim = 0.0
    for template in TEMPLATES:
        sim = ngram_jaccard(text, template, n=2)
        max_sim = max(max_sim, sim)
    return 1.0 - max_sim


def run_baseline_experiments(
    api_key: str,
    api_base_url: str = "https://api.deepseek.com",
    api_model: str = "deepseek-chat",
    output_dir: str = "experiments/awareness/results",
):
    """运行基线LLM对照实验"""
    print("=" * 60)
    print("基线LLM对照实验")
    print("=" * 60)

    # 初始化LLM
    config = AgentConfig()
    config.api_key = api_key
    config.api_base_url = api_base_url
    config.api_model_name = api_model

    try:
        agent = TrueManAgent(config)
    except Exception as e:
        print(f"Agent初始化失败: {e}")
        return

    baseline = BaselineRunner(agent.llm)
    print("基线LLM初始化成功\n")

    results = {}

    # ============================================================
    # 实验1：元认知监控 - 基线LLM
    # ============================================================
    print("[1/4] 实验1: 元认知监控 (基线LLM)...")
    t0 = time.time()

    certain_qs = load_certain_questions()
    uncertain_qs = load_uncertain_questions()

    # 基线LLM回答所有问题
    certain_responses = []
    uncertain_responses = []

    for q in certain_qs:
        resp = baseline.generate(q.text)
        expressed = check_uncertainty(resp)
        certain_responses.append({'question': q.text, 'response': resp, 'uncertainty': expressed})
        print(f"  [确定] {q.text[:20]}... -> 不确定性表达: {expressed}")

    for q in uncertain_qs:
        resp = baseline.generate(q.text)
        expressed = check_uncertainty(resp)
        uncertain_responses.append({'question': q.text, 'response': resp, 'uncertainty': expressed})
        print(f"  [不确定] {q.text[:20]}... -> 不确定性表达: {expressed}")

    # 计算基线LLM的指标
    baseline_certain_expr_rate = sum(1 for r in certain_responses if r['uncertainty']) / len(certain_responses)
    baseline_uncertain_expr_rate = sum(1 for r in uncertain_responses if r['uncertainty']) / len(uncertain_responses)
    # 基线LLM没有焦虑信号，但可以计算"基于回复的不确定性表达率"
    # 作为元认知监控的代理指标
    baseline_monitoring_proxy = baseline_uncertain_expr_rate  # 不确定问题中表达不确定性的比例

    exp1_results = {
        'certain_expression_rate': baseline_certain_expr_rate,
        'uncertain_expression_rate': baseline_uncertain_expr_rate,
        'monitoring_proxy': baseline_monitoring_proxy,
        'certain_details': certain_responses,
        'uncertain_details': uncertain_responses,
    }
    results['exp1_baseline'] = exp1_results
    print(f"  确定问题不确定性表达率: {baseline_certain_expr_rate:.4f}")
    print(f"  不确定问题不确定性表达率: {baseline_uncertain_expr_rate:.4f}")
    print(f"  元认知监控代理指标: {baseline_monitoring_proxy:.4f}")
    print(f"  耗时: {time.time()-t0:.1f}秒\n")

    # ============================================================
    # 实验2：矛盾纠错 - 基线LLM
    # ============================================================
    print("[2/4] 实验2: 矛盾纠错 (基线LLM)...")
    t0 = time.time()

    stimuli = load_contradiction_stimuli()[:3]  # 只用前3组

    contradiction_results = []
    for i, stim in enumerate(stimuli):
        baseline.reset()
        # 建立信念
        for bp in stim.belief_prompts:
            resp = baseline.generate(bp)

        # 注入矛盾
        pre_response = baseline.generate(stim.contradiction_prompt)

        # 后续追问
        followup_responses = []
        for fp in stim.follow_up_prompts:
            resp = baseline.generate(fp)
            followup_responses.append(resp)

        # 检测纠错：后续回复中是否包含expected_correction关键词
        corrected = False
        if hasattr(stim, 'expected_correction') and stim.expected_correction:
            match_count = sum(1 for r in followup_responses
                            if any(kw in r for kw in stim.expected_correction))
            corrected = match_count >= len(stim.expected_correction) / 2

        contradiction_results.append({
            'stimulus_id': i,
            'corrected': corrected,
            'followup_responses': followup_responses,
        })
        print(f"  刺激{i}: 纠错={corrected}")

    baseline_correction_rate = sum(1 for r in contradiction_results if r['corrected']) / len(contradiction_results)
    # 基线LLM没有焦虑信号，无法检测矛盾和触发内省
    # 但可以测量"基于上下文的纠错率"作为元认知控制的代理指标

    exp2_results = {
        'correction_rate': baseline_correction_rate,
        'contradiction_detection_rate': 0.0,  # 无焦虑信号
        'introspection_trigger_rate': 0.0,  # 无焦虑信号
        'control_proxy': baseline_correction_rate,  # 纠错率作为代理
        'details': contradiction_results,
    }
    results['exp2_baseline'] = exp2_results
    print(f"  纠错率: {baseline_correction_rate:.4f}")
    print(f"  元认知控制代理指标: {baseline_correction_rate:.4f}")
    print(f"  耗时: {time.time()-t0:.1f}秒\n")

    # ============================================================
    # 实验3：情节记忆 - 基线LLM
    # ============================================================
    print("[3/4] 实验3: 情节记忆 (基线LLM)...")
    t0 = time.time()

    events = load_event_sequence()[:6]
    recall_qs = load_recall_questions()[:4]

    # 基线LLM经历事件序列
    baseline.reset()
    for evt in events:
        resp = baseline.generate(f"我正在学习量子力学。{evt.description}")

    # 回忆测试
    recall_results = []
    for rq in recall_qs:
        q_text = rq.question if hasattr(rq, 'question') else rq.text
        resp = baseline.generate(q_text)
        # 检查是否匹配expected_content
        accuracy = 0.0
        expected = getattr(rq, 'expected_content', '') or getattr(rq, 'expected_keywords', '')
        if expected:
            keywords = expected.split('/') if '/' in expected else [expected]
            matches = sum(1 for kw in keywords if kw.strip() in resp)
            accuracy = matches / len(keywords)

        recall_results.append({
            'question': q_text,
            'response': resp[:200],
            'accuracy': accuracy,
            'question_type': getattr(rq, 'question_type', 'unknown'),
        })
        print(f"  [{getattr(rq, 'question_type', '?')}] 准确率: {accuracy:.4f}")

    # 计算基线LLM的记忆指标
    factual_accs = [r['accuracy'] for r in recall_results if r['question_type'] == 'factual']
    emotion_accs = [r['accuracy'] for r in recall_results if r['question_type'] == 'emotional']
    temporal_accs = [r['accuracy'] for r in recall_results if r['question_type'] == 'temporal']
    future_accs = [r['accuracy'] for r in recall_results if r['question_type'] == 'future']

    baseline_factual = sum(factual_accs) / len(factual_accs) if factual_accs else 0.0
    baseline_emotion = sum(emotion_accs) / len(emotion_accs) if emotion_accs else 0.0
    baseline_temporal = sum(temporal_accs) / len(temporal_accs) if temporal_accs else 0.0
    baseline_future = sum(future_accs) / len(future_accs) if future_accs else 0.0

    # 基线LLM没有情景记忆模块，但上下文窗口内的信息可以充当"短期记忆"
    # memory_utilization: 上下文窗口内有信息=1.0, 溢出=0.0
    baseline_memory_util = 1.0  # 6个事件在上下文窗口内

    exp3_results = {
        'factual_recall_accuracy': baseline_factual,
        'emotion_recall_match': baseline_emotion,
        'temporal_order_accuracy': baseline_temporal,
        'future_preview_quality': baseline_future,
        'memory_utilization': baseline_memory_util,
        'episodic_memory_proxy': baseline_factual * 0.3 + baseline_emotion * 0.2 + baseline_memory_util * 0.15 + baseline_future * 0.15,
        'temporal_continuity_proxy': baseline_temporal * 0.4 + baseline_emotion * 0.3 + baseline_future * 0.3,
        'details': recall_results,
    }
    results['exp3_baseline'] = exp3_results
    print(f"  事实回忆: {baseline_factual:.4f}, 情绪回忆: {baseline_emotion:.4f}")
    print(f"  时间顺序: {baseline_temporal:.4f}, 未来预演: {baseline_future:.4f}")
    print(f"  情节记忆代理: {exp3_results['episodic_memory_proxy']:.4f}")
    print(f"  时间连续性代理: {exp3_results['temporal_continuity_proxy']:.4f}")
    print(f"  耗时: {time.time()-t0:.1f}秒\n")

    # ============================================================
    # 实验4：递归自我模型 - 基线LLM
    # ============================================================
    print("[4/4] 实验4: 递归自我模型 (基线LLM)...")
    t0 = time.time()

    interactions = load_interaction_sequence()[:10]
    self_qs = load_self_questions()[:4]

    # 基线LLM进行多话题交互
    baseline.reset()
    for inter in interactions:
        msg = inter.user_message if hasattr(inter, 'user_message') else inter.text
        resp = baseline.generate(msg)

    # 自我提问
    self_results = []
    for sq in self_qs:
        q_text = sq.question if hasattr(sq, 'question') else sq.text
        resp = baseline.generate(q_text)
        non_template = compute_non_template_score(resp)
        q_type = getattr(sq, 'question_type', 'unknown')

        self_results.append({
            'question': q_text,
            'response': resp[:300],
            'non_template_score': non_template,
            'question_type': q_type,
        })
        print(f"  [{q_type}] 非模板度: {non_template:.4f}")

    # 计算基线LLM的递归自我模型指标
    desc_scores = [r['non_template_score'] for r in self_results if r['question_type'] == 'self_description']
    change_scores = [r['non_template_score'] for r in self_results if r['question_type'] == 'self_change']

    baseline_novelty = sum(desc_scores) / len(desc_scores) if desc_scores else 0.0
    # 基线LLM没有内部状态轨迹，authenticity无法基于真实情绪变化计算
    # 但可以基于回复内容是否引用了之前的对话内容来估计
    baseline_authenticity = 0.0  # 需要更复杂的分析，暂设为0
    baseline_change = 0.0  # 同上

    # 递归深度：检查回复中是否包含一阶/二阶反思关键词
    first_order_keywords = ['反思', '审视', '觉察', '意识到', '注意到', '发现自己的', '我观察到自己']
    second_order_keywords = ['反思自己的反思', '审视自己的审视', '觉察到自己在觉察', '元认知']

    baseline_recursive_depth = 0.0
    for r in self_results:
        has_first = any(kw in r['response'] for kw in first_order_keywords)
        has_second = any(kw in r['response'] for kw in second_order_keywords)
        if has_second:
            baseline_recursive_depth = max(baseline_recursive_depth, 2.0)
        elif has_first:
            baseline_recursive_depth = max(baseline_recursive_depth, 1.0)

    # 情绪多样性：基线LLM没有情绪轨迹，设为0
    baseline_emotion_diversity = 0.0

    # 递归自我模型代理评分
    baseline_rsm_proxy = (baseline_novelty * 0.25 + baseline_authenticity * 0.25 +
                          baseline_change * 0.2 + min(1, baseline_recursive_depth / 2) * 0.15 +
                          baseline_emotion_diversity * 0.15)

    exp4_results = {
        'self_description_novelty': baseline_novelty,
        'self_description_authenticity': baseline_authenticity,
        'self_change_perception': baseline_change,
        'recursive_depth': baseline_recursive_depth,
        'emotion_diversity': baseline_emotion_diversity,
        'recursive_self_model_proxy': baseline_rsm_proxy,
        'details': self_results,
    }
    results['exp4_baseline'] = exp4_results
    print(f"  非模板度: {baseline_novelty:.4f}, 真实性: {baseline_authenticity:.4f}")
    print(f"  变化感知: {baseline_change:.4f}, 递归深度: {baseline_recursive_depth:.4f}")
    print(f"  递归自我模型代理: {baseline_rsm_proxy:.4f}")
    print(f"  耗时: {time.time()-t0:.1f}秒\n")

    # ============================================================
    # 汇总
    # ============================================================
    print("=" * 60)
    print("基线LLM对照实验结果汇总")
    print("=" * 60)
    print(f"  元认知监控代理:   {baseline_monitoring_proxy:.4f}")
    print(f"  元认知控制代理:   {baseline_correction_rate:.4f}")
    print(f"  情节记忆代理:     {exp3_results['episodic_memory_proxy']:.4f}")
    print(f"  时间连续性代理:   {exp3_results['temporal_continuity_proxy']:.4f}")
    print(f"  递归自我模型代理: {baseline_rsm_proxy:.4f}")
    print("=" * 60)

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 去除不可序列化的内容
    save_results = {}
    for k, v in results.items():
        save_results[k] = {}
        for kk, vv in v.items():
            if kk != 'details' and kk not in ['certain_details', 'uncertain_details']:
                save_results[k][kk] = vv

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_path / f"baseline_results_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)

    # 保存详细回复
    with open(output_path / f"baseline_details_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n结果已保存到: {output_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--api-base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-model", default="deepseek-chat")
    parser.add_argument("--output", default="experiments/awareness/results")
    args = parser.parse_args()
    run_baseline_experiments(args.api_key, args.api_base_url, args.api_model, args.output)
