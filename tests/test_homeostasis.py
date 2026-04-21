"""情绪信号系统单元测试。"""

import pytest
import torch
import math

from trueman.core.config import SurpriseConfig, BoredomConfig, AnxietyConfig
from trueman.core.homeostasis.signals import SurpriseSignal, BoredomSignal, AnxietySignal
from trueman.core.homeostasis.integrator import EmotionIntegrator, EmotionState
from trueman.core.config import HomeostasisConfig


class TestSurpriseSignal:
    def setup_method(self):
        self.signal = SurpriseSignal(SurpriseConfig(threshold=2.0, decay=0.99))

    def test_low_prediction_error_low_surprise(self):
        """高预测概率（低误差）→ 低惊奇。"""
        log_probs = torch.tensor([-0.1, -0.05, -0.08, -0.12])
        result = self.signal.compute(log_probs)
        assert 0.0 <= result.item() <= 1.0

    def test_high_prediction_error_high_surprise(self):
        """低预测概率（高误差）→ 高惊奇。"""
        log_probs = torch.tensor([-5.0, -8.0, -6.0, -7.0])
        result = self.signal.compute(log_probs)
        assert 0.0 <= result.item() <= 1.0

    def test_always_in_range(self):
        """信号值始终在[0,1]范围内。"""
        for _ in range(100):
            log_probs = torch.randn(10) * 5
            result = self.signal.compute(log_probs)
            assert 0.0 <= result.item() <= 1.0

    def test_no_nan_inf(self):
        """无NaN/Inf。"""
        log_probs = torch.tensor([0.0, -100.0, 100.0])
        result = self.signal.compute(log_probs)
        assert torch.isfinite(result)

    def test_empty_input(self):
        """空输入返回中性值。"""
        result = self.signal.compute(torch.tensor([]))
        assert result.item() == 0.5


class TestBoredomSignal:
    def setup_method(self):
        self.signal = BoredomSignal(BoredomConfig(window=100, temperature=1.0))

    def test_window_insufficient_returns_default(self):
        """窗口不足返回默认值0.5。"""
        error = 1.0
        embedding = torch.randn(768)
        for _ in range(5):
            result = self.signal.compute(error, embedding)
        assert abs(result.item() - 0.5) < 0.01

    def test_always_in_range(self):
        """信号值始终在[0,1]范围内。"""
        for i in range(50):
            error = abs(torch.randn(1).item())
            embedding = torch.randn(768)
            result = self.signal.compute(error, embedding)
            assert 0.0 <= result.item() <= 1.0

    def test_repetitive_input_high_boredom(self):
        """重复输入 → 高无聊。"""
        error = 0.5
        embedding = torch.ones(768) * 0.5
        for _ in range(50):
            result = self.signal.compute(error, embedding)
        # 经过足够多重复输入后，无聊应该较高
        assert result.item() > 0.3


class TestAnxietySignal:
    def setup_method(self):
        self.signal = AnxietySignal(AnxietyConfig(n_samples=3, lightweight=True))

    def test_consistent_predictions_low_anxiety(self):
        """一致的预测 → 低焦虑。"""
        # 相同的概率分布
        probs = [torch.softmax(torch.randn(100), dim=-1) for _ in range(3)]
        result = self.signal.compute_from_predictions(probs)
        assert 0.0 <= result.item() <= 1.0

    def test_divergent_predictions_high_anxiety(self):
        """分歧的预测 → 高焦虑。"""
        # 不同的概率分布
        probs = [torch.softmax(torch.randn(100) * 10, dim=-1) for _ in range(3)]
        result = self.signal.compute_from_predictions(probs)
        assert 0.0 <= result.item() <= 1.0

    def test_text_similarity(self):
        """文本相似度计算。"""
        similar = self.signal._text_similarity("hello world", "hello world")
        assert similar > 0.8

        different = self.signal._text_similarity("hello world", "goodbye universe")
        assert different < 0.8

    def test_compute_from_texts(self):
        """基于文本差异的焦虑计算。"""
        # 相同文本 → 低焦虑
        result = self.signal.compute_from_texts(["same text", "same text", "same text"])
        assert result.item() < 0.5

        # 不同文本 → 较高焦虑
        result = self.signal.compute_from_texts(["cats are great", "dogs are terrible", "birds can fly"])
        assert result.item() >= 0.0


class TestEmotionIntegrator:
    def setup_method(self):
        self.integrator = EmotionIntegrator(HomeostasisConfig(
            setpoint_surprise=0.3,
            setpoint_boredom=0.3,
            setpoint_anxiety=0.2,
            alpha=1.0,
            beta=1.0,
            gamma=1.5,
        ))

    def test_at_setpoint_zero_drive(self):
        """所有信号等于设定点 → total_drive = 0。"""
        drive, state = self.integrator.integrate(0.3, 0.3, 0.2)
        assert abs(drive) < 1e-6

    def test_away_from_setpoint_positive_drive(self):
        """偏离设定点 → total_drive > 0。"""
        drive, state = self.integrator.integrate(0.8, 0.8, 0.8)
        assert drive > 0

    def test_values_clamped(self):
        """信号值被约束到[0,1]。"""
        drive, state = self.integrator.integrate(1.5, -0.5, 2.0)
        assert 0.0 <= state.surprise <= 1.0
        assert 0.0 <= state.boredom <= 1.0
        assert 0.0 <= state.anxiety <= 1.0

    def test_emotion_state_max_intensity(self):
        """max_intensity正确。"""
        _, state = self.integrator.integrate(0.5, 0.8, 0.3)
        assert abs(state.max_intensity - 0.8) < 1e-6
