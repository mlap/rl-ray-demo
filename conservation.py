
import torch
import gym
import gym_conservation
import gym_fishing
from ray import tune
from ray.rllib import agents
torch.cuda.device_count()


tune.register_env("conservation-v6", lambda config: NonStationaryV6(config))
tune.register_env("fishing-v1", lambda config: gym_fishing.envs.FishingCtsEnv(config))

# Custom environment creator utility
def env_creator(env_name):
    if env_name == 'fishing-v1':
        from gym_fishing.envs import FishingCtsEnv as env
    elif env_name == "conservation-v6":
        from gym_conservation.envs import NonStationaryV6 as env
    elif env_name == "fishing-v0":
        from gym_fishing.envs import FishingEnv as env
    else:
        raise NotImplementedError
    return env


# Use custom env creator instead of gym.make() to create callable env
# Then register env with label in ray tune.
env = env_creator('conservation-v6')
tune.register_env('conservation-v6', lambda config: env())
env1 = env_creator("fishing-v0")
tune.register_env('fishing-v0', lambda config: env1())
env1 = env_creator("fishing-v1")
tune.register_env('fishing-v1', lambda config: env1())

from gym_conservation.envs import NonStationaryV5 as consv5
tune.register_env('conservation-v5', lambda config: consv5())

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "conservation-v6",
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 2,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    "num_gpus": 1,
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
         "use_lstm": True,
#         "use_attention": True,
#        "fcnet_hiddens": [128, 128],
#        "fcnet_activation": "relu",
    },
    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    }
}

# Create our RLlib Trainer.
# trainer = agents.a3c.a2c.A2CTrainer(config=config)
#trainer = agents.ddpg.apex.ApexDDPGTrainer(config=config)
#trainer = agents.ppo.APPOTrainer(config=config)

#trainer = impala.ImpalaTrainer(config=config)


for _ in range(4):
    trainer.train()

out = trainer.evaluate() 
print(out)
