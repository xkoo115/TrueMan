"""最小化 CPU 端到端验证脚本。"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trueman.core.agent import TrueManAgent
from trueman.core.config import AgentConfig

config = AgentConfig()
config.base_model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
config.device = 'cpu'
config.load_in_4bit = False
config.load_in_8bit = False
config.memory_size = 200
config.awake_threshold = 5
config.anxiety.n_samples = 1
config.anxiety.lightweight = True
config.thresholds.anxiety_emergency_threshold = 0.99
config.thresholds.anxiety_introspection_threshold = 0.95
config.lora.rank = 4
config.lora.max_experts = 3
config.lora.target_modules = ['q_proj', 'v_proj']

print('初始化 Agent...')
agent = TrueManAgent(config)
agent.sleep.min_traces = 3
agent.sleep.nrem_steps = 10
agent.sleep.rem_steps = 2
print(f'OK hidden_size={agent.llm.hidden_size}')

# 3步交互（确保超过 min_traces=3）
for i, q in enumerate(['你好', '1+1等于几？', '中国的首都是哪里？']):
    t0 = time.time()
    response, emotion = agent.step(q)
    dt = time.time() - t0
    lora = agent.lora_pool.num_experts if agent.lora_pool else 0
    print(f'Step {i+1}: 惊奇={emotion.surprise:.3f} 无聊={emotion.boredom:.3f} 焦虑={emotion.anxiety:.3f} | 记忆={agent.episodic_memory.size} LoRA={lora} | {dt:.1f}s')

# 睡眠整合
print('触发睡眠整合...')
t0 = time.time()
expert_id = agent.force_sleep()
dt = time.time() - t0
lora = agent.lora_pool.num_experts if agent.lora_pool else 0
print(f'睡眠: expert_id={expert_id}, LoRA={lora}, {dt:.1f}s')

# 检查适配器
adapter_dir = 'adapters'
if os.path.exists(adapter_dir):
    experts = sorted([d for d in os.listdir(adapter_dir) if d.startswith('expert_')])
    print(f'适配器: {experts}')

print('DONE')
