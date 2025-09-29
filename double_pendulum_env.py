import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces

class DoublePendulumEnv(gym.Env):
    def __init__(self, render=True):
        super(DoublePendulumEnv, self).__init__()
        self._render = render
        if self._render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)


        self.robotId = p.loadURDF("double_pendulum.urdf", [0, 0, 0.1])

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.max_steps = 600
        self.current_step = 0
        self.upright_steps = 0
        p.setJointMotorControl2(self.robotId, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.robotId, 1, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.robotId, 2, p.VELOCITY_CONTROL, force=0)
        self.reset()

    def seed(self, seed=None):
        # This method is required for compatibility with some SB3 wrappers.
        pass

    def reset(self):
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 0.1], [0, 0, 0, 1])
        p.resetJointState(self.robotId, 0, 0)
        p.resetJointState(self.robotId, 1, np.pi)
        p.resetJointState(self.robotId, 2, 0)
        self.current_step = 0
        self.upright_steps = 0
        return self._get_obs()

    def step(self, action):
        p.setJointMotorControl2(self.robotId, 0, p.TORQUE_CONTROL, force=action[0] * 100)
        p.stepSimulation()

        obs = self._get_obs()
        
        x = obs[0]
        theta1 = obs[2]
        theta2 = obs[4]
        u = action[0]

        # The reward is a penalty for deviation from the upright position and for control effort.
        # The target is to have x=0, theta1=0, theta2=0.
        
        # Normalize angles to be in [-pi, pi] range for the reward calculation
        theta1_reward = (theta1 + np.pi) % (2 * np.pi) - np.pi
        theta2_reward = (theta2 + np.pi) % (2 * np.pi) - np.pi

        w_x = 0.1
        w_theta1 = 1.0
        w_theta2 = 2.0
        w_u = 0.01

        reward = np.exp(- (w_x * x**2 + w_theta1 * theta1_reward**2 + w_theta2 * theta2_reward**2 + w_u * u**2))

        self.current_step += 1
        done = False

        # Check for success condition
        angle_threshold = 0.15  # 5 degrees in radians
        is_upright = abs(theta1_reward) < angle_threshold and abs(theta2_reward) < angle_threshold

        if is_upright:
            self.upright_steps += 1
        else:
            self.upright_steps = 0

        if self.upright_steps >= 10:
            done = True
            reward += 100  # Bonus for achieving the goal
        
        if x > 2.4 or x < -2.4:
            done = True
            reward = -100.0 # Apply a large negative penalty
        
        if self.current_step >= self.max_steps:
            done = True

        return obs, reward, done, {}

    def _get_obs(self):
        cart_pos = p.getLinkState(self.robotId, 0)[0]
        # print("cart_pos: ", cart_pos)
        cart_vel, _ = p.getBaseVelocity(self.robotId)
        pole1_state = p.getJointState(self.robotId, 1)
        pole2_state = p.getJointState(self.robotId, 2)

        obs = np.array([
            cart_pos[0],
            cart_vel[0],
            pole1_state[0],
            pole1_state[1],
            pole2_state[0],
            pole2_state[1]
        ])
        return obs

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()