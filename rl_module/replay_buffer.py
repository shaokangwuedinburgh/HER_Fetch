import numpy as np
import copy

class replay_buffer(object):
    def __init__(self, env_param, args):
        self.env_params = env_param
        self.args = args

        self.max_buffer_size = args.buffer_size
        self.T = args.max_time_step  # 一次的最大仿真时间
        self.size = self.max_buffer_size // self.T
        self.current_size = 0  # 当前的episode位置
        self.is_full = False

        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }
        self.compute_reward_func = compute_reward


    def store_experience(self, episode_experience_dict):
        # episode_experience = [state, action, reward, next_state, done, goal]
        if self.current_size >= self.size:
            self.current_size = self.current_size % self.size
            self.is_full = True
        for key in self.buffer.keys():
            self.buffer[key][self.current_size] = episode_experience_dict[key]
        self.current_size += 1

    def sample_her_transitions(self, transitions, final_goal, sample_size):
        transitions_copy = copy.deepcopy(transitions)
        # 更新reward
        for i in range(sample_size):
            transitions_copy["reward"][i] = \
                self.compute_reward_func(
                    achieved_goal=transitions["ag"][i],
                    goal=final_goal[i],
                    info=None
                )
        return transitions_copy