import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import copy
from rl_module.actor import Actor
from rl_module.critic import Critic
from rl_module.replay_buffer import replay_buffer
from her_modules.her import her_sampler
from datetime import datetime
import logging
from torch.distributions import Normal
from mpi_utils.mpi_utils import sync_networks, sync_grads
from mpi_utils.normalizer import normalizer
from mpi4py import MPI

class DDPG(object):
    def __init__(self, env_param, args, environment):
        self.args = args
        self.env_param = env_param
        self.env = environment

        self.actor = Actor(env_param, args)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = Critic(env_param, args)
        self.target_critic = copy.deepcopy(self.critic)
        sync_networks(self.actor)
        sync_networks(self.critic)

        # self.replay_bufer = Replay_Buffer(max_buffer_size=replay_buffer_size)
        self.opt_actor = Adam(self.actor.parameters(), lr=args.lr_actor)
        self.opt_critic = Adam(self.critic.parameters(), lr=args.lr_critic)
        self.action_bound = env_param["action_max"]

        self.device = args.device
        # replay_strategy, replay_k, reward_func=None
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_param, self.args, self.her_module.sample_her_transitions)
        self.model_path = self.args.model_path

        self.o_norm = normalizer(size=env_param['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_param['goal'], default_clip_range=self.args.clip_range)



    def concat_state_goal(self, obs, g):

        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)

        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        return inputs

    def clip_obs_and_goal(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self.clip_obs_and_goal(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()



    def _select_actions(self, pi, success_rate):
        # add noise to the action to explore the environment
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_param['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_param['action_max'], self.env_param['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_param['action_max'], high=self.env_param['action_max'], \
                                           size=self.env_param['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)

        # the adaptive noise is added after the constant noise
        # def adaptive_noise(success_rate):
        #     q_success_rate = np.clip(self.args.policy_alpha - success_rate, self.args.policy_beta,
        #                              self.args.policy_alpha)
        #     mu, sigma = 0, 1
        #     probability = Normal(mu, sigma)
        #     noise = probability.sample([4])
        #     action_noise = q_success_rate * self.env_param["action_max"] * noise
        #     action_noise = action_noise.numpy()
        #     return action_noise
        #
        # action_noise = adaptive_noise(success_rate)
        # action += action_noise
        # action = np.clip(action, -self.env_param['action_max'], self.env_param['action_max'])

        return action

    def _soft_update_target_network(self, target_net, net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.args.tau) + param.data * self.args.tau)

    def learn(self):
        # start learning
        success_rate = 0
        for epoch in range(self.args.n_epoch):
            for episode in range(self.args.n_episode):
                ep_obs, ep_ag, ep_g, ep_action = [], [], [], []
                observation = self.env.reset()
                obs = observation["observation"]
                ag = observation["achieved_goal"]
                g = observation["desired_goal"]
                for t in range(self.args.max_time_step):
                    with torch.no_grad():
                        out = self.concat_state_goal(obs, g)
                        pi = self.actor(out)
                        action = self._select_actions(pi, success_rate)
                        # env requires action to be numpy type
                    observation, reward, done, info = self.env.step(action)
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    ep_g.append(g.copy())
                    ep_action.append(action.copy())

                    obs = observation["observation"]
                    ag = observation["achieved_goal"]

                ep_obs.append(obs)
                ep_ag.append(ag)
                trajectory = {"obs": np.array([ep_obs]),
                              "ag": np.array([ep_ag]),
                              "g": np.array([ep_g]),
                              "actions": np.array([ep_action])}
                self.buffer.store_episode(trajectory)
                self._update_normalizer([trajectory["obs"], trajectory["ag"], trajectory["g"], trajectory["actions"]])

                for _ in range(self.args.update_times):
                    self._update_network()

                if episode % self.args.update_times_target == 0:
                    self._soft_update_target_network(self.target_actor, self.actor)
                    self._soft_update_target_network(self.target_critic, self.critic)

            success_rate = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                torch.save([self.actor.state_dict()], self.model_path + '/model.pt')
    def _update_network(self):
        transitions = self.buffer.sample(self.args.batch_size)
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self.clip_obs_and_goal(o, g)
        transitions['obs_next'], transitions['g_next'] = self.clip_obs_and_goal(o_next, g)

        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)


        # transfer them into the tensor
        inputs_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        with torch.no_grad():
            actions_next = self.target_actor(inputs_next_tensor)
            q_next_value = self.target_critic(inputs_next_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)


        # the q loss
        real_q_value = self.critic(inputs_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor(inputs_tensor)
        actor_loss = -self.critic(inputs_tensor, actions_real).mean()

        # start to update the network
        self.opt_actor.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.opt_actor.step()

        # update the critic_network
        self.opt_critic.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.opt_critic.step()

    def _eval_agent(self):
        total_success_rate = []
        for _ in range(10):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.args.max_time_step):
                # self.env.render()
                with torch.no_grad():
                    input_tensor = self.concat_state_goal(obs, g)
                    pi = self.actor(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
