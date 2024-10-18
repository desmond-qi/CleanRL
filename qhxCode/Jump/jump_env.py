import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import glfw
from scipy.spatial.transform import Rotation as Rot
from gymnasium.envs.mujoco import MujocoEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

class HumanoidJumpEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],  # 声明支持的渲染模式
        "render_fps": 1000,
    }

    class robotInf:
        def __init__(self):
            self.initPos_euler = None
            self.initPos_quate = None
            self.jointUpperLimit = None
            self.jointLowerLimit = None

    def __init__(self, xml_file="jumping_robot.xml", render_mode=None, **kwargs):
        # 换算一下初始姿态角到四元数
        self.robotInf = self.robotInf()
        self.robotInf.initPos_euler = [0.0, 16.0 * D2R, 0.0]
        trans = Rot.from_euler('xyz', self.robotInf.initPos_euler)
        self.robotInf.initPos_quate = trans.as_quat()
        # 定义机器人的相关参数
        self.robotInf.jointUpperLimit = [ 2.0, 0.5, 2.5, 10.0*D2R, 90.0*D2R, 10.0*D2R, 10.0*D2R, 20.0*D2R, 120.0*D2R, 40.0*D2R, 20.0*D2R, 10.0*D2R, 20.0*D2R, 120.0*D2R, 40.0*D2R, 20.0*D2R,  90.0*D2R,  90.0*D2R]
        self.robotInf.jointLowerLimit = [-2.0,-0.5, 0.5,-10.0*D2R,-20.0*D2R,-10.0*D2R,-10.0*D2R,-90.0*D2R,   5.0*D2R,-55.0*D2R,-20.0*D2R,-10.0*D2R,-90.0*D2R,   5.0*D2R,-55.0*D2R,-20.0*D2R,-180.0*D2R,-180.0*D2R]

        # 定义动作空间和观测空间
        self.model_ = mujoco.MjModel.from_xml_path(xml_file)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.model_.nq + self.model_.nv + 6,), dtype=np.float32)

        # 使用 Mujoco 创建仿人机器人
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=1,
            observation_space=observation_space,
            render_mode=render_mode,
            default_camera_config={"trackbodyid": 2, "distance": 3.0, "lookat": np.array([0.0, 0.0, 1.15]), "elevation": -20.0},
            **kwargs
        )

    def reset_model(self):
        # 设置初始位置和速度
        qpos_initial = np.array([0.0, 0.0, 1.02] + list(self.robotInf.initPos_quate) + [0.0, -70.0*D2R, 100.0*D2R, -46.0*D2R, 0.0, 0.0, -70.0*D2R, 100.0*D2R, -46.0*D2R, 0.0, -40.0*D2R, -40.0*D2R])
        qvel_initial = np.zeros(self.model.nv)

        self.set_state(qpos_initial, qvel_initial)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = 3.0
        self.viewer.cam.lookat[:3] = [0.0, 0.0, 1.15]
        self.viewer.cam.elevation = -20.0

    def _get_obs(self):
        pos = self.data.qpos
        vel = self.data.qvel

        accel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "root_accel")
        gyro_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "root_gyro")

        accel_data = self.data.sensordata[accel_sensor_id:accel_sensor_id + 3]
        gyro_data = self.data.sensordata[gyro_sensor_id:gyro_sensor_id + 3]

        return np.concatenate([pos, vel, accel_data, gyro_data])
    
    def _calculate_reward(self):
        height = self.data.qpos[2]
        return height

    def _is_done(self):
        # 设定终止条件，例如摔倒或者高度低于某个值
        height = self.data.qpos[2]
        trans = Rot.from_quat([self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6]])
        eulerPos = trans.as_euler('xyz')
        if height < 0.5 or abs(eulerPos[0]) > 5.0 * D2R or eulerPos[1] > 90.0 * D2R or eulerPos[1] < -10.0 * D2R or abs(eulerPos[2]) > 5.0 * D2R :
            return True
        # if height < 0.5:
        #     return True
        return False

    def step(self, action):
        # 使用 Mujoco 的模拟步进接口
        self.do_simulation(action, self.frame_skip)
        # 获取新的观测值
        obs = self._get_obs()
        # 计算奖励
        reward = self._calculate_reward()
        # 判断是否结束 
        done = self._is_done()
        # 返回当前步的结果
        return obs, reward, done, False, {}


from gymnasium.envs.registration import register

register(
    id='QhxJump-v0',
    entry_point='jump_env:HumanoidJumpEnv',
)