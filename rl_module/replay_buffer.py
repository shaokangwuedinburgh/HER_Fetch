import numpy as np
import copy

class replay_buffer(object):
    def __init__(self, env_param, args, sample_func):
        self.env_params = env_param
        self.args = args

        self.max_buffer_size = args.buffer_size
        self.T = args.max_time_step  # 一次的最大仿真时间
        self.size = int(self.max_buffer_size // self.T)
        self.current_size = 0  # 当前的episode位置
        self.is_full = False

        # why the obs and ag size is self.T + 1?
        # cause we need it to calculate next_obs and next ag
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }
        self.sample_func = sample_func


    def store_episode(self, trajectory):
        # episode_experience_dict is the trajectory in this current episode
        if self.current_size >= self.size:
            self.current_size = self.current_size % self.size
            self.is_full = True
        for key in self.buffers.keys():
            self.buffers[key][self.current_size] = trajectory[key]
        self.current_size += 1

    def sample(self, batch_size):
        temp_buffer = {}

        # if the buffer is full, then we can sample all of them
        # otherwise, we can only sample from 0 to self.current_size
        if not self.is_full:
            for key in self.buffers.keys():
                temp_buffer[key] = self.buffers[key][:self.current_size]

        else:
            for key in self.buffers.keys():
                temp_buffer[key] = self.buffers[key]

        temp_buffer["obs_next"] = self.buffers["obs"][:, 1:, :]
        temp_buffer["ag_next"] = self.buffers["ag"][:, 1:, :]
        transtions = self.sample_func(temp_buffer, batch_size)

        return transtions