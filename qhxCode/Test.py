import gymnasium as gym
import Qhx_dummy_env  # 确保加载模块以执行注册

env = gym.make('QhxDummy-v0')
print("Environment created successfully!")
