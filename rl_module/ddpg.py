import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import copy
from rl_module.actor import Actor
from rl_module.critic import Critic
from rl_module.replay_buffer import replay_buffer
from her_modules.her import her_sampler


class DDPG(object):
    def __init__(self, env_param, args, environment):
        self.args = args
        self.env_param = env_param
        self.env = environment

        self.actor = Actor(env_param, args)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = Critic(env_param, args)
        self.target_critic = copy.deepcopy(self.critic)

        # self.replay_bufer = Replay_Buffer(max_buffer_size=replay_buffer_size)
        self.opt_actor = Adam(self.actor.parameters(), lr=args.lr_actor)
        self.opt_critic = Adam(self.critic.parameters(), lr=args.lr_critic)
        self.action_bound = env_param["action_max"]

        self.device = args.device
        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)
        # replay_strategy, replay_k, reward_func=None
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)


    def sample(self, sample_size):
        samples = self.replay_bufer.sample(sample_size)
        return samples

    def choose_action(self, observation, goal):
        # state is a numpy array
        state = np.concatenate([observation, goal])
        staet_concat_goal = torch.DoubleTensor(state).reshape(1, -1).to(self.device)

        action = self.actor(staet_concat_goal).detach().reshape(
            -1)  # + torch.normal(mean=0., std=0.1, size=(1,))
        # action = action.clamp(-self.action_bound, self.action_bound)
        action = action.cpu().numpy()
        action += 0.3 * 1 * np.random.randn(*action.shape)
        action = np.clip(action, -1, 1)
        # random actions...
        random_actions = np.random.uniform(low=-1, high=1, \
                                           size=4)
        # choose if use the random actions
        action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)
        return action

    def learn(self, transition):
        # start learning
        for update_times in range(1):
            # 先更新 critic
            state = torch.from_numpy(transition["obs"]).to(self.device)
            action = torch.from_numpy(transition["actions"]).to(self.device)
            reward = torch.from_numpy(transition["reward"]).to(self.device)
            next_state = torch.from_numpy(transition["next_obs"]).to(self.device)
            goal = torch.from_numpy(transition["goal"]).to(self.device)
            state_concat_goal = torch.concat([state, goal], dim=1).to(self.device)

            with torch.no_grad():
                next_state_concat_goal = torch.concat([next_state, goal], dim=1).to(self.device)
                next_action = self.target_actor(next_state_concat_goal)
                target = reward + GAMMA * self.target_critic(next_state_concat_goal, next_action)
            estimate = self.critic(state_concat_goal, action)
            loss_q = F.mse_loss(estimate, target)
            self.opt_critic.zero_grad()
            loss_q.backward()
            self.opt_critic.step()

            # 再更新 actor
            actor_action = self.actor(state_concat_goal)
            goal = -torch.mean(self.critic(state_concat_goal, actor_action))
            self.opt_actor.zero_grad()
            goal.backward()
            self.opt_actor.step()

    def soft_update(self, target_net, net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)





def save_torch_model(model, path):
    torch.save(model.state_dict(), path)
