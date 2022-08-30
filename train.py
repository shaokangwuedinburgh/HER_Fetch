import mujoco_py
import gym
import numpy as np
import random
import torch
import gym_robotics.envs as environment
from arguments import get_args
from rl_module.ddpg import DDPG

def get_env_params(env):
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              }
    return params

def launch(args):
    env = environment.FetchReachEnv()
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_params = get_env_params(env)
    ddpg_trainer = DDPG(env_params, args, env)
    # ddpg_trainer.learn()


if __name__ == '__main__':
    args = get_args()
    launch(args)
