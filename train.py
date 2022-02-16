"""
Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search to try different learning rates

You can visualize experiment results in ~/ray_results using TensorBoard.

Run example with defaults:
$ python custom_env.py
For CLI options:
$ python custom_env.py --help
"""
import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print

import gym_conservation
import gym_fishing

## rllib ignores gym registered names, need to register manually:
## note these envs were not written to take a single parameter dictionary ("config")
tune.register_env("conservation-v6", lambda config: gym_conservation.envs.NonStationaryV6())
tune.register_env("fishing-v1", lambda config: gym_fishing.envs.FishingCtsEnv())

from gym_conservation.envs import NonStationaryV6
from gym_fishing import envs as fishing

os.environ["RLLIB_NUM_GPUS"] = "1"

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)

parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=50, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=7, help="Reward at which we stop training."
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init(local_mode=args.local_mode)


    config = {
        "env": "fishing-v1",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 2,  # parallelism
        "framework": "torch",
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    # automated run with Tune and grid search and TensorBoard
    print("Training automatically with Ray Tune")
    results = tune.run(args.run, config=config, stop=stop)

    if args.as_test:
        print("Checking if learning goals were achieved")
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
