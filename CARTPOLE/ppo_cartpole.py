import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import gym

from matplotlib import pyplot as plt

print("Importing Tensorflow")
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical 
import keras.layers as layers

from tqdm import trange


total_steps = 1_000_000 # number of total steps taken before ending learning 
T = 500         # number of steps before learning
clip_coef = 0.2 # PPO clip parameter
N = 5          # max number of updates before accumulating more experience
delta = 0.02    # max KL divergence allowed between old and new policy

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


class Agent():
    def __init__(self, env):
        
        self.actor = tf.keras.Sequential()
        self.actor.add(layers.Input(shape=( np.array(env.observation_space.shape).prod(), )))
        self.actor.add(layers.Dense(32, activation='linear'))
        self.actor.add(layers.Dense(16, activation='linear'))
        self.actor.add(layers.Dense(16, activation='linear'))
        self.actor.add(layers.Dense(env.action_space.n, activation='softmax'))
        
        self.actor.build()
        self.actor.summary()
        
        self.critic = tf.keras.Sequential()      
        self.critic.add(layers.Input(shape=( np.array(env.observation_space.shape).prod(), )))
        self.critic.add(layers.Dense(32, activation='relu'))
        self.critic.add(layers.Dense(16, activation='relu'))
        self.critic.add(layers.Dense(1, activation='relu'))
        
        self.critic.build()
        self.critic.summary()
        
    def get_value(self, state):
        return self.critic( np.array([state]) )[0]

    def get_action(self, state):
        st = np.array([state])
        logits = self.actor( [st] )[0]
        probs = Categorical(logits=logits)
        
        action = probs.sample()
        
        return action, probs.prob(action)
    
    def get_trainable_variables_actor(self):
        return self.actor.trainable_variables

    def get_trainable_variables_critic(self):
        return self.critic.trainable_variables
    
    def train_model(self, states, advantages, values, actions, old_prob):
        n = 1
        while True: # do while
            
            # Compute loss and optimize policy
            with tf.GradientTape() as tape:
                
                # PPO loss
                y_pred = self.actor([states])
                prob_sampler = Categorical(logits=y_pred) 
                prob = prob_sampler.prob(actions)
            
                ratio = prob / old_prob
                clip_ratio = tf.clip_by_value(ratio, 1 - clip_coef, 1 + clip_coef)

                surrogate1 = ratio * advantages
                surrogate2 = clip_ratio * advantages

                ppo_loss = - tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                
            grad = tape.gradient(ppo_loss, self.actor.trainable_variables)
            optimizer.apply_gradients(zip(grad, self.actor.trainable_variables))
        
            with tf.GradientTape() as tape:
                
                # MSE loss
                y_pred_crit = self.critic([states])
                # advantage + value = return, which we want crit to predict
                mse_loss = tf.reduce_mean( tf.pow(y_pred_crit - advantages+values[:-1], 2) )
                
            grad = tape.gradient(mse_loss, self.critic.trainable_variables)
            optimizer.apply_gradients(zip(grad, self.critic.trainable_variables))
            
            # estimate KL divergence for the break condition
            approx_kl = tf.math.reduce_mean( prob * tf.math.log(ratio) )
            # (estimation only since true value would mean sampling all possible actions)
            if n >= N:
                break
            if approx_kl > delta:
                print(f"stopped on KL divergence (iteration {n})")
                break
            n += 1
            
def get_advantages(rewards, valeurs, done, gamma=0.99, lmbda=0.95):
    
    advantages = np.zeros(shape=(len(rewards),))
    
    def delta(t):
        # don't add next value if done is true
        return rewards[t] + gamma*valeurs[t+1]*(done[t]==False) - valeurs[t]
    
    # accumulate in last, reset if done is True
    last = 0
    for t in reversed(range(len(done))):
        advantages[t] = last = delta(t) + (gamma*lmbda)*last*(done[t]==False)

    return advantages

print("Creating Environment")
env = gym.make("CartPole-v1")

print("Creating Agent")
agent = Agent(env) # policy Pi

env.action_space.seed(0)

next_done = False

mean_episode_lengths = []
min_episode_lengths = []
max_episode_lengths = []

print("Creating Environment")
for i in range(total_steps // T):
    print(f"\n==== Epoch {i+1}/{total_steps // T} ====")
    
    # Utilisation de la policy T fois
    observations = np.zeros((T,) + env.observation_space.shape)
    rewards      = np.zeros((T,))
    dones        = np.zeros((T,))
    actions      = np.zeros((T,) + env.action_space.shape)
    values       = np.zeros((T+1,))
    probs        = np.zeros((T,))
    
    next_obs = env.reset()
    current_episode_steps = 0
    
    min_length = 1000
    max_length = 0
    mean_length = 0
    nb_finished = 0
    print('== Stepping ==')
    for step in trange(0, T):
        env.render()
        obs = next_obs
        done = next_done

        action, prob = agent.get_action( obs )
        value = agent.get_value( obs )
        
        next_obs, reward, next_done, info = env.step( np.array(action) ) # step in N environments
        
        observations[step] = obs
        rewards[step]      = reward
        dones[step]        = done
        actions[step]      = action
        values[step]       = value
        probs[step]        = prob
        
        if next_done:
            # print(f"Finished episode in {current_episode_steps} steps")
            min_length = min(min_length, current_episode_steps)
            max_length = max(max_length, current_episode_steps)
            mean_length += current_episode_steps
            nb_finished += 1
            current_episode_steps = 0
            next_obs = env.reset()
        else:
            current_episode_steps += 1


    # adding the expected value of the last observation (no action taken)
    values[T] = agent.get_value(next_obs)
    
    mean_length /= nb_finished
    mean_episode_lengths.append(mean_length)
    min_episode_lengths.append(min_length)
    max_episode_lengths.append(max_length)
    plt.clf()
    plt.fill_between(range(len(max_episode_lengths)), max_episode_lengths, min_episode_lengths, color=(0.85, 0.85, 0.85))
    plt.plot(range(len(mean_episode_lengths)), mean_episode_lengths)
    plt.draw()
    plt.pause(0.01)
    
    print('\n== Training ==')
    
    # Calcul des avantages At
    advantages = get_advantages(rewards, values, dones)
    
    agent.train_model(observations, advantages, values, actions, probs)

env.close()
