import torch
import torch.nn as nn
import torch.nn.functional as F

from config import args


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, action_dim)

        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=args.device)


    def forward(self, state):

        a = F.relu(self.linear1(state))
        a = F.relu(self.linear2(a))

        return self.max_action * torch.tanh(self.linear3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(state_dim + action_dim, 256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, 1)


    def forward(self, state, action):

        s_a = torch.cat([state, action], dim=-1)

        q1 = F.relu(self.linear1(s_a))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        q2 = F.relu(self.linear4(s_a))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)

        return q1, q2


    def forward_Q1(self, state, action):

        s_a = torch.cat([state, action], dim=-1)

        q1 = F.relu(self.linear1(s_a))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        return q1




