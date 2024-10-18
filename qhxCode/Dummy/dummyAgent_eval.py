import gymnasium as gym
import torch
import numpy as np
from Qhx_ppo import Agent  # 导入训练时使用的 Agent 类

# 创建评估环境
eval_env = gym.make("QhxDummy-v0", render_mode="rgb_array")

# 创建模型实例并加载训练好的模型
model_name = "dummy_model_2024-10-12_16-17-35.pth"
model_path = f"model/{model_name}"
# 手动添加 single_observation_space 和 single_action_space 属性
eval_env.single_observation_space = eval_env.observation_space
eval_env.single_action_space = eval_env.action_space

# 创建模型实例并加载保存的模型参数
model = Agent(eval_env)  # 修改这里，传入观测和动作空间
model.load_state_dict(torch.load(model_path))  # 加载保存的模型权重
model.eval()  # 设置模型为评估模式

# 创建用于录制视频的评估环境
eval_env = gym.wrappers.RecordVideo(eval_env, f"videos/eval_{model_name}", episode_trigger=lambda t: t == 0)

# 重置环境
obs, _ = eval_env.reset()
done = False

# 测试智能体表现
while not done:
    # 选择训练完成后的智能体的动作
    obs_tensor = torch.Tensor(obs).unsqueeze(0)  # 将观测值转换为张量，并增加一个批次维度
    with torch.no_grad():  # 禁用梯度计算
        action, _, _, _ = model.get_action_and_value(obs_tensor)
    action = action.cpu().numpy()[0]
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated

# 关闭评估环境，确保视频被正确保存
eval_env.close()
