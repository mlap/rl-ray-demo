

import torch
import gym
import gym_conservation
import gym_fishing
from ray import tune
from ray.rllib import agents
from ray.rllib.agents.ppo import PPOTrainer
torch.cuda.device_count()


# In[17]:


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


# In[18]:


# Use custom env creator instead of gym.make() to create callable env
# Then register env with label in ray tune.
env = env_creator('conservation-v6')
tune.register_env('conservation-v6', lambda config: env())
env1 = env_creator("fishing-v0")
tune.register_env('fishing-v0', lambda config: env1())
env1 = env_creator("fishing-v1")
tune.register_env('fishing-v1', lambda config: env1())


# In[19]:


## minimal example
#import gym, ray
#from ray.rllib.agents.ppo import PPOTrainer
#trainer = PPOTrainer(env="fishing-v1", config={})
#trainer.train()


# In[14]:


# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "fishing-v1",
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 4,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    "num_gpus": 1,
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

# Create our RLlib Trainer.
trainer = PPOTrainer(config=config)


# In[23]:


for _ in range(10):
    trainer.train()


# In[24]:



# Evaluate the trained Trainer (and render each timestep to the shell's output).
trainer.evaluate()

