import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

print("Importing Tensorflow")
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical 
from tf_agents.distributions.masked import MaskedCategorical

import keras.layers as layers

from time import sleep

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    partial_obs=False,
    max_steps=2000,
    render_theme=2,
    ai2s=[
        #microrts_ai.coacAI,
        #microrts_ai.lightRushAI,
        #microrts_ai.workerRushAI,
        microrts_ai.POHeavyRush,
    ],
    map_paths=["maps/8x8/basesWorkers8x8.xml"],
    reward_weight=np.array([20.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)

nb_games = 4

size_map = 8
size_map2 = 64


class Agent():
    def __init__(self, envs):
        
        self.actor = tf.keras.models.load_model('./model_saves/actor_8x8.tf')
        
        self.actor.summary()
        
    @tf.function
    def get_action(self, state, mask):
        
        logits = self.actor( state )
        
        # reduce to a flat vec (envs, size_map2, action_space_noncompressed_flat)
        grid_logits = logits.reshape(-1, size_map2, envs.action_plane_space.nvec.sum())
        grid_env_mask = mask.reshape(-1, size_map2, envs.action_plane_space.nvec.sum())
        
        # split at each length of action space (6, 4, 4, 4, 4, 7, 49)
        # returns a list of tensors of size (7, envs, size_map2, 6 or 4 or 7 or 49)
        split_logits = tf.split(grid_logits,   envs.action_plane_space.nvec.tolist(), axis=2)
        split_mask   = tf.split(grid_env_mask, envs.action_plane_space.nvec.tolist(), axis=2)

        # create sampler for each action space with its respective mask
        multi_categoricals = [
            MaskedCategorical(logits=logits, mask=mask, neg_inf=-1e8)
            for (logits, mask) in zip(split_logits, split_mask)
        ]
        
        # sample each action space
        action = tf.stack([
            categorical.sample() 
            for categorical in multi_categoricals
        ])
        prob = tf.stack([
            prob_sampler.log_prob(action[i])
            for (prob_sampler, i) in zip(multi_categoricals, range(len(multi_categoricals)))
        ])
        
        # rearrange into (envs, size_map2, 7)
        action = tf.transpose(action, [1, 2, 0])
        prob   = tf.transpose(prob,   [1, 2, 0])
        
        return action, tf.reduce_sum(prob, axis=(1, 2))

agent = Agent(envs)

for i in range(nb_games):
    print(f"\n==== Game {i+1}/{nb_games} ====")
    
    next_done = np.zeros((envs.num_envs,))
    next_obs  = envs.reset()
    envs.render()
    
    sleep(1.5)
    

    while next_done[0] == 0:
        
        envs.render()
        sleep(0.005)
        obs = next_obs
        done = next_done

        action_mask = envs.get_action_mask()
        action, prob = agent.get_action( obs, action_mask )
        
        next_obs, reward, next_done, info = envs.step( np.array(action) ) # step in N environments
        
    if reward[0] > 0:
        print("Game won!")
    if reward[0] < 0:
        print("Game lost :(")
    if reward[0] == 0:
        print("It's a draw!")
    
envs.close()