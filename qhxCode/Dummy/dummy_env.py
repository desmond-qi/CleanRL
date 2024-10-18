import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class DummyEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],  # 声明支持的渲染模式
    }

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
        self.target_position = 2.0  # 目标位置
        self.target_velocity = .0   # 目标速度

        # 最大步数设置
        self.max_steps = 2000
        self.current_step = 0

        # 设置渲染相关
        self.render_mode = render_mode
        if render_mode in ["human", "rgb_array"]:
            if render_mode == "human":
                plt.ion()  # 使 matplotlib在后台不关闭窗口
            self.fig, self.ax = plt.subplots()
            self.particle, = plt.plot([], [], 'bo', markersize=10)

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
            force = -0.5
        elif action == 1:
            force = 0.5
        
        # 更新动力学
        dt = 0.1  # 时间步
        acceleration = force / self.mass
        velocity += acceleration * dt
        position += velocity * dt

        # 更新状态
        self.state = np.array([position, velocity], dtype=np.float32)

        # 计算奖励
        distance_to_target = np.abs(self.target_position - position)
        reward = -distance_to_target  # 基础奖励，鼓励接近目标位置
        if distance_to_target < 0.1:
            reward += 10  # 如果接近目标位置，给予额外奖励
        reward -= 0.01 * np.abs(acceleration)  # 减小加速度惩罚的权重

        # 判断是否达到目标
        terminated = np.abs(self.target_position - position) < 0.1 and np.abs(self.target_velocity - velocity)  # 达到目标位置即自然结束
        truncated = position < -15.0 or position > 15.0 or np.abs(velocity) > 100.0  # 可以增加一个时间步数的限制以判断是否截断

        return self.state, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            print(f"Position: {self.state[0]}, Velocity: {self.state[1]}")
        elif self.render_mode == "rgb_array":
            # 使用 matplotlib 渲染
            self.ax.clear()
            self.ax.set_xlim(-15, 15)
            self.ax.set_ylim(-1, 1)
            self.ax.grid(True, linestyle='--')
            self.ax.set_xlabel('Position')
            self.ax.set_title('Single Particle Environment')
            self.particle, = self.ax.plot(self.state[0], 0, 'bo', markersize=10)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()  # 刷新绘图

            # 将图像渲染为 RGB 数组
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            width, height = self.fig.canvas.get_width_height()
            return image.reshape(height, width, 3)

    def close(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)

from gymnasium.envs.registration import register

register(
    id='QhxDummy-v0',
    entry_point='dummy_env:DummyEnv',
)