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

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

T = 800
N = 5
total_steps = 10_000_000
clip_coef = 0.2
delta = 0.03
optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4, epsilon=1e-5)
load_from_files = False

envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=6,
    partial_obs=False,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI, microrts_ai.lightRushAI, microrts_ai.workerRushAI, microrts_ai.coacAI, microrts_ai.lightRushAI, microrts_ai.workerRushAI],
    map_paths=["maps/8x8/basesWorkers8x8.xml"],
    reward_weight=np.array([20.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)

num_predicted_parameters = len(envs.action_plane_space.nvec)
size_map = 8
size_map2 = 64

losses = []

class Agent():
    def __init__(self, envs):
        
        if load_from_files:
            self.actor = tf.keras.models.load_model('./actor.tf')
            self.actor.summary()
            
            self.critic = tf.keras.models.load_model('./critic.tf')
            self.critic.summary()
            
        else:
        
            inp = layers.Input( shape = envs.observation_space.shape )
            x = layers.Conv2D(32, (3, 3), padding='same', activation='linear')(inp)
            x = layers.MaxPool2D((3, 3), (2, 2), padding='same')(x)
            x = layers.ReLU()(x)
            
            x = layers.Conv2D(64, (3, 3), padding='same', activation='linear')(x)
            x = layers.MaxPool2D((3, 3), (2, 2), padding='same')(x)
            out = layers.ReLU()(x)
            
            x_act = layers.Conv2DTranspose(32, (3, 3), (2, 2), padding='same', activation='relu')(out)
            x_act = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x_act)
            out_act = layers.Conv2DTranspose(78, (3, 3), (2, 2), padding='same', activation='linear')(x_act)
            
            out_cri = layers.Flatten()(out)
            out_cri = layers.Dense(128, activation='linear')(out_cri)
            out_cri = layers.Dense(1, activation='linear')(out_cri)
            
            self.actor = tf.keras.Model(inputs=inp, outputs=out_act)
            self.actor.build(input_shape=envs.observation_space.shape)
            self.actor.summary()
            
            self.critic = tf.keras.Model(inputs=inp, outputs=out_cri)        
            self.critic.build(input_shape=envs.observation_space.shape)
            self.critic.summary()
        
    def get_value(self, state):
        return self.critic( state ).reshape(-1) # (envs, 1) -> (envs,)

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
    
    def train_model(self, states, advantages, values, actions, old_prob, mask):
        n = 1
        
        actions = tf.transpose(actions, [3, 0, 1, 2])
        
        while True:
            
            ppo_loss, mse_loss, approx_kl = self.train_once(states, advantages, values, actions, old_prob, mask)
            
            if n >= N:
                break
            if approx_kl > delta:
                print(f"stopped on KL divergence (iteration {n}/{N})")
                break
            n += 1
            
        agent_loss = tf.math.reduce_mean(ppo_loss).numpy()
        critic_loss = tf.math.reduce_mean(mse_loss).numpy()
        print("agent_loss:", agent_loss)
        print("critic_loss:", critic_loss)
        
        losses.append(agent_loss)
            
        plt.clf()
        plt.plot(range(len(losses)), losses)
        plt.draw()
        plt.pause(0.001)
    
    @tf.function
    def train_once(self, states, advantages, values, actions, old_prob, mask):
        with tf.GradientTape() as tape:
                
            # MSE Loss for critic
            y_critic = self.critic( states.reshape(-1, size_map, size_map, 27) )
            y_critic = y_critic.reshape( -1, envs.num_envs )
            mse_loss = tf.reduce_mean( tf.pow(y_critic - advantages+values[:-1], 2) )
            
            # PPO Loss for actor
            y_pred = self.actor( states.reshape(-1, size_map, size_map, 27) )
            y_pred = y_pred.reshape( T, -1, size_map2, envs.action_plane_space.nvec.sum() )
            mask   = mask.reshape( T, -1, size_map2, envs.action_plane_space.nvec.sum() )
            
            # split at each length of action space (6, 4, 4, 4, 4, 7, 49)
            split_logits = tf.split( y_pred, envs.action_plane_space.nvec.tolist(), axis=3)
            split_mask   = tf.split( mask,   envs.action_plane_space.nvec.tolist(), axis=3)

            # create sampler for each action space with its respective mask
            multi_categoricals = [
                MaskedCategorical(logits=logits, mask=mask, neg_inf=-1e8)
                for (logits, mask) in zip(split_logits, split_mask)
            ]
            
            # get action prob distribution
            prob = tf.stack([
                prob_sampler.log_prob(actions[i])
                for (prob_sampler, i) in zip(multi_categoricals, range(len(multi_categoricals)))
            ])
            prob = tf.transpose( prob,   [1, 2, 3, 0])
            prob = tf.reduce_sum( prob, axis=(2, 3))
            
            # ratio based on log probability
            ratio = tf.exp( prob - old_prob )
            clip_ratio = tf.clip_by_value(ratio, 1 - clip_coef, 1 + clip_coef)
            surrogate1 = ratio * advantages
            surrogate2 = clip_ratio * advantages

            ppo_loss = - tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            
            full_loss = ppo_loss + 0.5*mse_loss # add mse for critic loss

        # Get grad and optimize models
        grad = tape.gradient(full_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        optimizer.apply_gradients(zip(grad, self.actor.trainable_variables + self.critic.trainable_variables))

        # compute approximate KL divergence (break condition)
        approx_kl = tf.math.reduce_mean( tf.math.exp(prob) * (prob - old_prob) )
        
        return ppo_loss, mse_loss, approx_kl
        
    def save_model(self):
        self.actor.save("actor.tf")
        self.critic.save("critic.tf")

def get_advantages(rewards, valeurs, done, gamma=0.99, lmbda=0.95):
    advantages = np.zeros(shape=rewards.shape)
    
    def delta(t):
        return rewards[t] + gamma*valeurs[t+1]*(done[t]==False) - valeurs[t]
    
    last = 0
    for t in reversed(range(done.shape[0])):
        advantages[t] = last = delta(t) + (gamma*lmbda)*last*(done[t]==False)
    
    return advantages

print("Creating Environment")
agent = Agent(envs)

next_done = np.zeros((envs.num_envs,))
next_obs  = envs.reset()
for i in range(total_steps // T):
    print(f"\n==== Epoch {i+1}/{total_steps // T} ====")
    
    # Utilisation de la policy T fois
    observations = np.zeros( (T,   envs.num_envs) + envs.observation_space.shape )
    rewards      = np.zeros( (T,   envs.num_envs) )
    dones        = np.zeros( (T,   envs.num_envs) )
    actions      = np.zeros( (T,   envs.num_envs, size_map2, num_predicted_parameters) )
    values       = np.zeros( (T+1, envs.num_envs) )
    probs        = np.zeros( (T,   envs.num_envs) )
    masks        = np.zeros( (T,)+ envs.get_action_mask().shape )
    
    current_episode_steps = 0
    print('== Stepping ==')

    for step in tqdm(range(0, T)):
        envs.render()
        
        obs = next_obs
        done = next_done

        action_mask = envs.get_action_mask()
        action, prob = agent.get_action( obs, action_mask )
        value = agent.get_value( obs )
        
        next_obs, reward, next_done, info = envs.step( np.array(action) ) # step in N environments
        
        observations[step] = obs
        rewards[step]      = reward
        dones[step]        = done
        actions[step]      = action
        values[step]       = value
        probs[step]        = prob
        masks[step]        = action_mask

    # adding the expected value of the last observation (no action taken)
    values[T] = agent.get_value(next_obs)
    
    # Calcul des avantages At
    advantages = get_advantages(rewards, values, dones)
    
    print('\n== Training ==')
    agent.train_model(observations, advantages, values, actions, probs, masks)
    
    agent.save_model()

envs.close()
