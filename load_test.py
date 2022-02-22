import torch
import gym
import gym_fishing
import ray
from ray import tune
import ray.rllib.agents.ppo as ppo

ray.init()
tune.register_env("fishing-v1", lambda config: gym_fishing.envs.FishingCtsEnv())

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "fishing-v1",
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 1,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    "num_gpus": 0,
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    "evaluation_interval": 2,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    }
}

model = ppo.PPOTrainer(config=config)
model.restore("trash/checkpoint_000010/checkpoint-10")
model.evaluate()
