"""
Example grid search tuning for some of our custom gym environments

You can visualize experiment results in ~/ray_results using TensorBoard.

Run example :
$ python train.py --run TD3

For CLI options:
$ python custom_env.py --help
"""
import argparse
import gym
import numpy as np
import os

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print

import gym_conservation
import gym_fishing

## rllib ignores gym registered names, need to register manually:
## note these envs were not written to take a single parameter dictionary ("config")
tune.register_env("conservation-v6", lambda config: gym_conservation.envs.NonStationaryV6())
tune.register_env("conservation-v5", lambda config: gym_conservation.envs.NonStationaryV5())
tune.register_env("fishing-v0", lambda config: gym_fishing.envs.FishingEnv())
tune.register_env("fishing-v1", lambda config: gym_fishing.envs.FishingCtsEnv())


os.environ["RLLIB_NUM_GPUS"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--env", type=str, default="fishing-v1", help="The gym environment to use."
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=500, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=3000000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=200, help="Reward at which we stop training."
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
        ## parameters to tune: BE SURE TO MATCH THESE TO ALGO
        ## PPO
         "lr": tune.loguniform(1e-5, 1e-2),
         "vf_clip_param": tune.uniform(10, 100),
        ## Fixed config settings: not tuned
        "env": args.env,
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
