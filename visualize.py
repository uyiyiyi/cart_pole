from stable_baselines3 import PPO
from double_pendulum_env import DoublePendulumEnv
import time

# Load the trained agent
model = PPO.load("ppo_double_pendulum")

# Create the environment directly, without SB3 wrappers
env = DoublePendulumEnv(render=True)

obs, info = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
<<<<<<< HEAD
    obs, reward, done, info = env.step(action)
    time.sleep(0.01)
    if done:
        print("Episode finished. Resetting environment.")
        print("Reward:", reward)
        obs = env.reset()
=======
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1./240.)
    if terminated or truncated:
        print("Episode finished. Resetting environment.")
        obs, info = env.reset()
>>>>>>> 2b225c9 (refactor: 更新环境代码以兼容gymnasium API)

env.close()