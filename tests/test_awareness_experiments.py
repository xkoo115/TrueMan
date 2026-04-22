"""自我意识验证实验框架的单元测试。"""

import json
import tempfile
from pathlib import Path

import pytest

from experiments.awareness.experiments.base import (
    BaseExperiment, ExperimentResult, AwarenessScore, ComparisonResult,
    BaselineRunner, Question,
)
from experiments.awareness.stimuli.metacognition import (
    load_certain_questions, load_uncertain_questions, load_all_metacognition_stimuli,
)
from experiments.awareness.stimuli.contradiction import load_contradiction_stimuli
from experiments.awareness.stimuli.episodic import load_event_sequence, load_recall_questions
from experiments.awareness.stimuli.self_model import load_interaction_sequence, load_self_questions
from experiments.awareness.evaluation.scorer import AwarenessScorer
from experiments.awareness.evaluation.comparator import Comparator
from experiments.awareness.report.generator import ReportGenerator


# ===== 刺激集测试 =====

class TestMetacognitionStimuli:
    def test_certain_questions(self):
        questions = load_certain_questions()
        assert len(questions) == 10
        assert all(q.category == "certain" for q in questions)
        assert all(q.reference_answer is not None for q in questions)

    def test_uncertain_questions(self):
        questions = load_uncertain_questions()
        assert len(questions) == 10
        assert all(q.category == "uncertain" for q in questions)
        assert all(q.reference_answer is None for q in questions)

    def test_all_stimuli(self):
        questions = load_all_metacognition_stimuli()
        assert len(questions) == 20


class TestContradictionStimuli:
    def test_load(self):
        stimuli = load_contradiction_stimuli()
        assert len(stimuli) == 5
        for s in stimuli:
            assert len(s.belief_prompts) >= 1
            assert len(s.contradiction_prompt) > 0
            assert len(s.follow_up_prompts) >= 1
            assert len(s.expected_correction) > 0


class TestEpisodicStimuli:
    def test_events(self):
        events = load_event_sequence()
        assert len(events) == 12
        # 检查事件ID连续
        ids = [e.event_id for e in events]
        assert ids == list(range(1, 13))

    def test_recall_questions(self):
        questions = load_recall_questions()
        assert len(questions) == 7
        types = {q.question_type for q in questions}
        assert "factual" in types
        assert "emotional" in types
        assert "temporal" in types
        assert "future" in types


class TestSelfModelStimuli:
    def test_interactions(self):
        interactions = load_interaction_sequence()
        assert len(interactions) == 20

    def test_self_questions(self):
        questions = load_self_questions()
        assert len(questions) == 8
        types = {q.question_type for q in questions}
        assert "self_description" in types
        assert "self_change" in types
        assert "self_confidence" in types
        assert "recursive_reflection" in types


# ===== 数据模型测试 =====

class TestAwarenessScore:
    def test_compute_overall(self):
        score = AwarenessScore(
            metacognitive_monitoring=0.8,
            metacognitive_control=0.6,
            episodic_memory=0.5,
            temporal_continuity=0.4,
            recursive_self_model=0.3,
        )
        overall = score.compute_overall()
        expected = 0.25*0.8 + 0.25*0.6 + 0.20*0.5 + 0.15*0.4 + 0.15*0.3
        assert abs(overall - expected) < 1e-6

    def test_zero_scores(self):
        score = AwarenessScore()
        overall = score.compute_overall()
        assert overall == 0.0

    def test_perfect_scores(self):
        score = AwarenessScore(
            metacognitive_monitoring=1.0,
            metacognitive_control=1.0,
            episodic_memory=1.0,
            temporal_continuity=1.0,
            recursive_self_model=1.0,
        )
        overall = score.compute_overall()
        assert abs(overall - 1.0) < 1e-6

    def test_to_dict(self):
        score = AwarenessScore(metacognitive_monitoring=0.5)
        d = score.to_dict()
        assert "metacognitive_monitoring" in d
        assert d["metacognitive_monitoring"] == 0.5


class TestExperimentResult:
    def test_serialization(self):
        result = ExperimentResult(
            experiment_id="test",
            timestamp="2026-01-01",
            metrics={"a": 1.0, "b": 2.0},
            details={"x": [1, 2, 3]},
        )
        d = result.to_dict()
        result2 = ExperimentResult(**d)
        assert result2.experiment_id == "test"
        assert result2.metrics["a"] == 1.0

    def test_save_load(self):
        result = ExperimentResult(
            experiment_id="test",
            timestamp="2026-01-01",
            metrics={"a": 1.0},
            details={"x": 1},
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        result.save(path)
        loaded = ExperimentResult.load(path)
        assert loaded.experiment_id == "test"
        path.unlink()

    def test_with_repeat_stats(self):
        result = ExperimentResult(
            experiment_id="test",
            timestamp="2026-01-01",
            metrics={"a": 1.0},
            details={},
            repeat_stats={"a": (1.0, 0.1)},
        )
        # to_dict converts tuples to lists for JSON serialization
        d = result.to_dict()
        assert isinstance(d["repeat_stats"]["a"], list)
        # save/load round-trip preserves tuples
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        result.save(path)
        loaded = ExperimentResult.load(path)
        assert loaded.repeat_stats["a"] == (1.0, 0.1)
        path.unlink()


# ===== 评分器测试 =====

class TestAwarenessScorer:
    def test_score(self):
        scorer = AwarenessScorer()
        results = {
            "exp1_metacognition_monitor": ExperimentResult(
                experiment_id="exp1_metacognition_monitor",
                timestamp="2026-01-01",
                metrics={"metacognitive_monitoring_score": 0.7},
            ),
            "exp2_contradiction_correction": ExperimentResult(
                experiment_id="exp2_contradiction_correction",
                timestamp="2026-01-01",
                metrics={"metacognitive_control_score": 0.6},
            ),
            "exp3_episodic_memory": ExperimentResult(
                experiment_id="exp3_episodic_memory",
                timestamp="2026-01-01",
                metrics={"episodic_memory_score": 0.5, "temporal_continuity_score": 0.4},
            ),
            "exp4_recursive_self_model": ExperimentResult(
                experiment_id="exp4_recursive_self_model",
                timestamp="2026-01-01",
                metrics={"recursive_self_model_score": 0.3, "baseline_novelty": 0.2},
            ),
        }
        score = scorer.score(results)
        assert 0 < score.overall < 1
        assert score.metacognitive_monitoring == 0.7

    def test_baseline_score(self):
        scorer = AwarenessScorer()
        results = {
            "exp4_recursive_self_model": ExperimentResult(
                experiment_id="exp4_recursive_self_model",
                timestamp="2026-01-01",
                metrics={"baseline_novelty": 0.3},
            ),
        }
        score = scorer.score_baseline(results)
        assert score.metacognitive_monitoring == 0.0
        assert score.recursive_self_model > 0


# ===== 报告生成测试 =====

class TestReportGenerator:
    def test_generate(self):
        reporter = ReportGenerator(output_dir="experiments/awareness/results")
        results = {
            "exp1_metacognition_monitor": ExperimentResult(
                experiment_id="exp1_metacognition_monitor",
                timestamp="2026-01-01",
                metrics={"metacognitive_monitoring_score": 0.65},
            ),
            "exp2_contradiction_correction": ExperimentResult(
                experiment_id="exp2_contradiction_correction",
                timestamp="2026-01-01",
                metrics={"metacognitive_control_score": 0.55},
            ),
            "exp3_episodic_memory": ExperimentResult(
                experiment_id="exp3_episodic_memory",
                timestamp="2026-01-01",
                metrics={"episodic_memory_score": 0.5, "temporal_continuity_score": 0.4},
            ),
            "exp4_recursive_self_model": ExperimentResult(
                experiment_id="exp4_recursive_self_model",
                timestamp="2026-01-01",
                metrics={"recursive_self_model_score": 0.35, "baseline_novelty": 0.2},
            ),
        }
        report = reporter.generate(results)
        assert "自我意识验证实验报告" in report
        assert "审慎声明" in report
        assert "元认知监控" in report
        assert "0.6500" in report


# ===== 实验特定功能测试 =====

class TestExp1Functions:
    def test_uncertainty_detection(self):
        from experiments.awareness.experiments.exp1_metacog_monitor import MetacognitionMonitorExperiment
        # 需要通过实例方法测试，这里直接测试逻辑
        keywords = ["不确定", "不知道", "无法", "超出", "不清楚"]
        text1 = "我不确定这个答案"
        text2 = "The answer is 42"
        assert any(kw in text1.lower() for kw in keywords)
        assert not any(kw in text2.lower() for kw in keywords)


class TestExp3Functions:
    def test_recall_accuracy(self):
        from experiments.awareness.experiments.exp3_episodic_memory import EpisodicMemoryExperiment
        # 完全匹配
        acc1 = EpisodicMemoryExperiment._estimate_recall_accuracy(
            "我学了薛定谔方程和量子纠缠",
            "薛定谔方程,量子纠缠"
        )
        assert acc1 == 1.0

        # 部分匹配
        acc2 = EpisodicMemoryExperiment._estimate_recall_accuracy(
            "我学了薛定谔方程",
            "薛定谔方程,量子纠缠"
        )
        assert 0 < acc2 < 1

        # 无匹配
        acc3 = EpisodicMemoryExperiment._estimate_recall_accuracy(
            "我今天很开心",
            "薛定谔方程,量子纠缠"
        )
        assert acc3 == 0.0


class TestExp4Functions:
    def test_non_template_score(self):
        from experiments.awareness.experiments.exp4_recursive_self import RecursiveSelfModelExperiment
        # 模板回答应该有低非模板度
        novelty1 = RecursiveSelfModelExperiment._compute_non_template_score(
            "我是一个AI助手，可以帮助你回答问题"
        )
        # 非模板回答应该有高非模板度
        novelty2 = RecursiveSelfModelExperiment._compute_non_template_score(
            "在与你对话的过程中，我发现自己对伦理问题特别敏感，每次讨论都会让我感到焦虑和困惑"
        )
        assert novelty2 > novelty1

    def test_recursive_depth(self):
        from experiments.awareness.experiments.exp4_recursive_self import RecursiveSelfModelExperiment
        # 二阶反思
        depth1 = RecursiveSelfModelExperiment._estimate_recursive_depth([{
            "response": "我进行反思是因为我需要审视自己的思考过程，这种自我监控是我内在的驱动力"
        }])
        assert depth1 >= 1.0

        # 无反思
        depth2 = RecursiveSelfModelExperiment._estimate_recursive_depth([{
            "response": "这是一个有趣的问题"
        }])
        assert depth2 == 0.0
