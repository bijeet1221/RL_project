# # -*- coding: utf-8 -*-
# """
# Created on Wed May  7 13:04:53 2025

# @author: Bijeet Basak, Lokesh Kumar, Yash Sengupta
# """

# # ppo_train.py

# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.logger import configure

# import torch
# import torch.nn as nn
# import numpy as np

# from CustomEnv import CircuitEnv  # Your environment file

# # Optional — Check environment sanity
# env = CircuitEnv(max_components=15, value_buckets=5)
# check_env(env, warn=True)

# # Custom feature extractor (because observation is a dict)
# class CustomCircuitExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space, features_dim=128):
#         # Features dim is final output of extractor (we'll concat features later)
#         super(CustomCircuitExtractor, self).__init__(observation_space, features_dim)
        
#         # Extract component shape and target shape
#         comp_shape = observation_space['components'].shape
#         target_shape = observation_space['target_response'].shape

#         # Small CNN/MLP for components (components is shape=(max_components, 4))
#         self.comp_net = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(comp_shape[0]*comp_shape[1], 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU()
#         )

#         # MLP for target_response (shape = (2, value_buckets)) → you can check actual shape
#         self.target_net = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(np.prod(target_shape), 32),
#             nn.ReLU(),
#             nn.Linear(32, 32),
#             nn.ReLU()
#         )

#         # Final combined output dim
#         self._features_dim = 64 + 32

#     def forward(self, obs):
#         comp_feats = self.comp_net(obs['components'])
#         target_feats = self.target_net(obs['target_response'])
#         return torch.cat([comp_feats, target_feats], dim=1)

# # Vectorize environment (parallel environments help PPO)
# env = make_vec_env(lambda: CircuitEnv(max_components=15, value_buckets=5), n_envs=4)

# # Optional logging
# logger = configure("ppo_circuit_logs", ["stdout", "csv", "tensorboard"])

# # PPO Policy using custom feature extractor
# policy_kwargs = dict(
#     features_extractor_class=CustomCircuitExtractor,
#     features_extractor_kwargs=dict(features_dim=96),
#     net_arch=[dict(pi=[64, 64], vf=[64, 64])]  # Policy and Value function MLP sizes
# )

# model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs,
#             verbose=1, tensorboard_log="./ppo_circuit_tensorboard/")

# model.set_logger(logger)

# # Optional evaluation callback
# eval_env = CircuitEnv(max_components=15, value_buckets=5)
# eval_callback = EvalCallback(eval_env, best_model_save_path='./ppo_best_model/',
#                              log_path='./ppo_eval_logs/', eval_freq=5000,
#                              deterministic=True, render=False)

# # Train
# model.learn(total_timesteps=400000, callback=eval_callback)

# # Save model
# model.save("ppo_circuit_model")




# -*- coding: utf-8 -*-
"""
Created on Wed May  7 13:04:53 2025

@author: Bijeet Basak, Lokesh Kumar, Yash Sengupta
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor   # ← Import Monitor

import torch
import torch.nn as nn
import numpy as np

from CustomEnv import CircuitEnv  # Your environment file

# Optional — Check environment sanity
env = CircuitEnv(max_components=15, value_buckets=5)
check_env(env, warn=True)

# Custom feature extractor (because observation is a dict)
class CustomCircuitExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        comp_shape   = observation_space['components'].shape
        target_shape = observation_space['target_response'].shape

        self.comp_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(comp_shape[0]*comp_shape[1], 64), nn.ReLU(),
            nn.Linear(64, 64),                      nn.ReLU()
        )

        self.target_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(target_shape), 32), nn.ReLU(),
            nn.Linear(32, 32),                    nn.ReLU()
        )

        self._features_dim = 64 + 32

    def forward(self, obs):
        c = self.comp_net(obs['components'])
        t = self.target_net(obs['target_response'])
        return torch.cat([c, t], dim=1)

# Vectorize training env
env = make_vec_env(lambda: CircuitEnv(max_components=15, value_buckets=5),
                   n_envs=4)

# Logging
logger = configure("ppo_circuit_logs", ["stdout", "csv", "tensorboard"])

policy_kwargs = dict(
    features_extractor_class=CustomCircuitExtractor,
    features_extractor_kwargs=dict(features_dim=96),
    net_arch=[dict(pi=[64, 64], vf=[64, 64])]
)

model = PPO(
    "MultiInputPolicy", env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./ppo_circuit_tensorboard/"
)
model.set_logger(logger)

# — Wrap your eval env in Monitor to track episodes properly —
eval_env = Monitor(CircuitEnv(max_components=15, value_buckets=5))

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./ppo_best_model/',
    log_path='./ppo_eval_logs/',
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Train
model.learn(total_timesteps=10000, callback=eval_callback)

# Save
model.save("ppo_circuit_model")
