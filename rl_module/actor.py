import torch.nn as nn

# pi(action | [state, goal])
class Actor(nn.Module):
    def __init__(self, env_param, args):
        super(Actor, self).__init__()
        self.state_space = env_param["obs"] + env_param["goal"]
        self.action_space = env_param["action"]
        self.bound = env_param["action_max"]
        self.relu = nn.ReLU()
        self.hidden_dim = args.hidden_dim
        self.fc1 = nn.Linear(self.state_space, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.action_space)
        self.tanh = nn.Tanh()

    def forward(self, state_concat_goal):
        out = self.relu(self.fc1(state_concat_goal))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.tanh(self.fc4(out))
        out = self.bound * out
        return out
