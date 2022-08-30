import torch.nn as nn
import torch


# Q([state, goal], action)
class Critic(nn.Module):
    def __init__(self, env_param, args):
        super(Critic, self).__init__()
        self.state_space = env_param["obs"] + env_param["goal"]
        self.action_space = env_param["action"]
        self.hidden_dim = args.hidden_dim
        self.bound = env_param["action_max"]
        self.fc1 = nn.Linear(self.state_space + self.action_space, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state_concat_goal, action):
        x = torch.cat([state_concat_goal, action / self.bound], dim=1)
        x = self.relu(self.fc1(x))  # out -> hidden dim
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.output(x)
        return x
