import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class DummyEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(DummyEnv, self).__init__()
        # 定义状态空间：质点的位置和速度
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        # 定义动作空间：施加的力
        self.action_space = spaces.Discrete(2)
        # 定义质点质量
        self.mass = 1.0

        # 质点的初始位置和速度
        self.state = None
        self.target_position = 10.0  # 目标位置

        # 设置渲染相关
        self.fig, self.ax = plt.subplots()
        self.particle, = plt.plot([], [], 'bo', markersize=10)

        if render_mode == "human" or render_mode == "rgb_array":
            self.fig, self.ax = plt.subplots()
            self.particle, = self.ax.plot([], [], 'bo', markersize=10) 

    def reset(self, seed=None, options=None):
        # 如果提供了随机种子，则设置种子
        if seed is not None:
            np.random.seed(seed)

        # 初始化状态
        self.state = np.array([np.random.uniform(-5.0, 5.0), 0.0], dtype=np.float32)

        # 返回初始状态，Gymnasium 需要返回 (obs, info)
        return self.state, {}
    
    def step(self, action):
        position, velocity = self.state
        # 根据动作选择施加的力
        if action == 0:
            force = -1.0
        elif action == 1:
            force = 1.0
        
        # 更新动力学
        dt = 0.1  # 时间步
        acceleration = force / self.mass
        velocity += acceleration * dt
        position += velocity * dt

        # 更新状态
        self.state = np.array([position, velocity], dtype=np.float32)

        # 计算奖励
        reward = -np.abs(self.target_position - position)

        # 判断是否达到目标
        terminated = np.abs(self.target_position - position) < 0.1  # 达到目标位置即自然结束
        truncated = False  # 可以增加一个时间步数的限制以判断是否截断

        return self.state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        if self.render_mode == "human":
            print(f"Position: {self.state[0]}, Velocity: {self.state[1]}")
        elif self.render_mode == "rgb_array":
            # 使用 matplotlib 渲染
            self.ax.clear()
            self.ax.set_xlim(-15, 15)
            self.ax.set_ylim(-1, 1)
            self.ax.set_xlabel('Position')
            self.ax.set_title('Single Particle Environment')
            self.particle, = self.ax.plot(self.state[0], 0, 'bo', markersize=10)
            self.fig.canvas.draw()

            # 将图像渲染为 RGB 数组
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            width, height = self.fig.canvas.get_width_height()
            return image.reshape(height, width, 3)

    def close(self):
        if hasattr(self, 'fig'):
            import matplotlib.pyplot as plt
            plt.close(self.fig)

from gymnasium.envs.registration import register

register(
    id='QhxDummy-v0',
    entry_point='Qhx_dummy_env:DummyEnv',
)


print("Environment QhxDummy-v0 has been registered.")
