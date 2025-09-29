from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from double_pendulum_env import DoublePendulumEnv

if __name__ == '__main__':
    # Configure the logger to output to a file
    # This will create a `log.txt` file in the current directory
    configure(folder=".", format_strings=["stdout", "log", "csv"])

    # Create the vectorized environment, and wrap each environment with a Monitor
    env = make_vec_env(DoublePendulumEnv, n_envs=8, vec_env_cls=SubprocVecEnv, wrapper_class=Monitor, env_kwargs={'render': False})

    # Instantiate the agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=1000000)

    # Save the agent
    model.save("ppo_double_pendulum")

    env.close()