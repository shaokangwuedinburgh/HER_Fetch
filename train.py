import mujoco_py
import gym
import numpy as np
import random
import torch
from mpi4py import MPI
import gym_robotics.envs as environment
from arguments import get_args
from rl_module.ddpg import DDPG
import os

def get_env_params(env):
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              }
    return params

def launch(args):
    env = environment.FetchPushEnv()
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    env_params = get_env_params(env)
    ddpg_trainer = DDPG(env_params, args, env)
    ddpg_trainer.learn()


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    args = get_args()
    launch(args)
