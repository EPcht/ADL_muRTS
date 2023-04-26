# https://www.youtube.com/watch?v=MEt6rrxH8W4&ab_channel=Weights%26Biases

import argparse
import os
from distutils.util import strtobool
import time
import random
import numpy as np
import torch
import gym
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    # parser.add_argument("--num-envs", type=int, default=4,
    #     help="the number of parallel game environments")
    # parser.add_argument("--num-steps", type=int, default=128,
    #     help="the number of steps to run in each environment per policy rollout")
    # parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Toggle learning rate annealing for policy and value networks")
    # parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Use GAE for advantage computation")
    # parser.add_argument("--gamma", type=float, default=0.99,
    #     help="the discount factor gamma")
    # parser.add_argument("--gae-lambda", type=float, default=0.95,
    #     help="the lambda for the general advantage estimation")
    # parser.add_argument("--num-minibatches", type=int, default=4,
    #     help="the number of mini-batches")
    # parser.add_argument("--update-epochs", type=int, default=4,
    #     help="the K epochs to update the policy")
    # parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Toggles advantages normalization")
    # parser.add_argument("--clip-coef", type=float, default=0.2,
    #     help="the surrogate clipping coefficient")
    # parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    # parser.add_argument("--ent-coef", type=float, default=0.01,
    #     help="coefficient of the entropy")
    # parser.add_argument("--vf-coef", type=float, default=0.5,
    #     help="coefficient of the value function")
    # parser.add_argument("--max-grad-norm", type=float, default=0.5,
    #     help="the maximum norm for the gradient clipping")
    # parser.add_argument("--target-kl", type=float, default=None,
    #     help="the target KL divergence threshold")
    args = parser.parse_args()
    # args.batch_size = int(args.num_envs * args.num_steps)
    # args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writter = SummaryWriter(f"run/{run_name}")
    writter.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" %("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    for i in range(100):
        writter.add_scalar("test_loss", i*2, global_step=i)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, "videos")
    observation = env.reset()
    for _ in range(200):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
            print(f"episodic return: {info['episode']['r']}")
    env.close()