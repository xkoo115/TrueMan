"""集成测试：验证Agent完整交互循环。"""

import pytest
import torch

from trueman.core.config import (
    AgentConfig, SurpriseConfig, BoredomConfig, AnxietyConfig, HomeostasisConfig,
)
from trueman.core.homeostasis.signals import SurpriseSignal, BoredomSignal, AnxietySignal
from trueman.core.homeostasis.integrator import EmotionIntegrator, EmotionState
from trueman.core.world_model.predictor import WorldModel
from trueman.core.environment import (
    DialogEnvironment, Observation, Action,
    ObservationType, ActionType,
)
from trueman.core.homeostasis.multiscale import (
    MultiScaleHomeostasis, Timescale,
)


class TestWorldModel:
    def setup_method(self):
        self.model = WorldModel(state_dim=128, hidden_dim=64)

    def test_forward(self):
        state = torch.randn(128)
        predicted = self.model(state)
        assert predicted.shape == (128,)

    def test_prediction_error(self):
        state = torch.randn(128)
        next_state = torch.randn(128)
        error = self.model.prediction_error(state, next_state)
        assert error.item() >= 0

    def test_online_update(self):
        state = torch.randn(128)
        next_state = torch.randn(128)
        loss_before = self.model.prediction_error(state, next_state).item()
        for _ in range(10):
            self.model.update(state, next_state, weight=1.0)
        loss_after = self.model.prediction_error(state, next_state).item()
        assert loss_after < loss_before


class TestDialogEnvironment:
    def setup_method(self):
        self.env = DialogEnvironment()

    def test_set_input_and_observe(self):
        self.env.set_input("hello")
        obs = self.env.observe()
        assert obs.content == "hello"
        assert obs.type == ObservationType.TEXT

    def test_execute_action(self):
        action = Action(type=ActionType.TEXT, content="hi there")
        feedback = self.env.execute(action)
        assert feedback.observation.content == "hi there"

    def test_reset(self):
        self.env.set_input("hello")
        self.env.reset()
        obs = self.env.observe()
        assert obs.content == ""


class TestMultiScaleHomeostasis:
    def setup_method(self):
        self.msh = MultiScaleHomeostasis()

    def test_update_rates(self):
        rate_ultra = self.msh.get_update_rate(Timescale.ULTRA_FAST)
        rate_fast = self.msh.get_update_rate(Timescale.FAST)
        rate_medium = self.msh.get_update_rate(Timescale.MEDIUM)
        rate_slow = self.msh.get_update_rate(Timescale.SLOW)
        assert rate_ultra > rate_fast > rate_medium > rate_slow

    def test_surprise_updates_fast(self):
        for _ in range(10):
            val = self.msh.update("surprise", 0.8)
        assert val > 0.5

    def test_boredom_updates_slowly(self):
        for _ in range(10):
            val = self.msh.update("boredom", 0.8)
        assert val < 0.8

    def test_should_trigger(self):
        for _ in range(100):
            self.msh.update("surprise", 0.9)
        assert self.msh.should_trigger("surprise", 0.9, threshold=0.5)


class TestEmotionSignalIntegration:
    def test_emotion_values_always_valid(self):
        """情绪信号值始终有效。"""
        surprise = SurpriseSignal(SurpriseConfig())
        boredom = BoredomSignal(BoredomConfig())
        integrator = EmotionIntegrator(HomeostasisConfig())

        for i in range(50):
            log_probs = torch.randn(10) * 3
            s = surprise.compute(log_probs)
            b = boredom.compute(abs(torch.randn(1).item()), torch.randn(768))

            drive, state = integrator.integrate(s.item(), b.item(), 0.0)
            assert 0.0 <= state.surprise <= 1.0
            assert 0.0 <= state.boredom <= 1.0
            assert 0.0 <= state.anxiety <= 1.0
            assert state.drive >= 0
