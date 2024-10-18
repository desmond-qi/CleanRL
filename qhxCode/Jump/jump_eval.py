import torch
import gymnasium as gym
from jump_env import HumanoidJumpEnv
import numpy as np

# 加载训练后的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model/jump_model_2024-10-17_21-21-45.pth"  # 这里要使用你保存的模型文件路径
state_dict = torch.load(model_path, map_location=device)

# 创建与训练时相同的模型结构
from jump_ppo import Agent
env = HumanoidJumpEnv(xml_file="/home/desmond/Desmond/CleanRL/qhxCode/Jump/jumping_robot.xml", render_mode=None)
env.single_observation_space = env.observation_space
env.single_action_space = env.action_space
trained_agent = Agent(env).to(device)
trained_agent.load_state_dict(state_dict)
trained_agent.eval()

# 创建测试环境
env = HumanoidJumpEnv(xml_file="/home/desmond/Desmond/CleanRL/qhxCode/Jump/jumping_robot.xml", render_mode="human")
obs, _ = env.reset()
obs = torch.tensor(obs, dtype=torch.float32).to(device)

done = False
cumulative_reward = 0.0

while not done:
    # 获取动作
    with torch.no_grad():
        action, _, _, _ = trained_agent.get_action_and_value(obs.unsqueeze(0))
    # 调整动作的维度
    action = action.cpu().numpy().squeeze()
    # 环境步进
    obs, reward, done, _, _ = env.step(action)
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    env.render()

    cumulative_reward += reward

print(f"Cumulative reward for the episode: {cumulative_reward}")
