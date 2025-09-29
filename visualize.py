from stable_baselines3 import PPO
from double_pendulum_env import DoublePendulumEnv
import time

# Load the trained agent
model = PPO.load("ppo_double_pendulum")

# Create the environment directly, without SB3 wrappers
env = DoublePendulumEnv(render=True)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    time.sleep(0.01)
    if done:
        print("Episode finished. Resetting environment.")
        print("Reward:", reward)
        obs = env.reset()

env.close()