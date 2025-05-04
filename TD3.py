import torch
import torch.nn.functional as F
import numpy as np


from config import args
from model import Actor , Critic
from replay_buffer import ReplayBuffer


class TD3:

    def __init__(self, max_action: np.ndarray):

        self.train_steps = 0

        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=args.device)
        self.policy_noise = args.policy_noise * self.max_action
        self.policy_noise_clip = args.policy_noise_clip * self.max_action
        self.gamma = torch.tensor(args.gamma, dtype=torch.float32, device=args.device)
        self.tau = torch.tensor(args.tau, dtype=torch.float32, device=args.device)


    def train(
            self,
            actor: Actor,
            actor_target: Actor,
            critic: Critic,
            critic_target: Critic,
            replay_buffer: ReplayBuffer,
            actor_optimizer: torch.optim.Adam,
            critic_optimizer: torch.optim.Adam
        ):

        self.train_steps += 1


        replays = replay_buffer.sample()

        states = torch.stack([replay.state for replay in replays])
        actions = torch.stack([replay.action for replay in replays])
        rewards = torch.stack([replay.reward for replay in replays])
        next_states = torch.stack([replay.next_state for replay in replays])
        not_dones = torch.stack([replay.not_done for replay in replays])


        # 計算 target_Q
        with torch.no_grad():

            next_actions = actor_target(next_states)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.policy_noise_clip , self.policy_noise_clip)
            next_actions = (next_actions + noise).clamp(-self.max_action , self.max_action)

            next_Q1s , next_Q2s = critic_target(next_states, next_actions)
            next_Qs = torch.min(next_Q1s, next_Q2s)
            target_Qs = rewards + not_dones * self.gamma * next_Qs


        # 計算 Q1 , Q2
        Q1s , Q2s = critic(states, actions)

        # MSE loss
        critic_loss = F.mse_loss(Q1s, target_Qs) + F.mse_loss(Q2s, target_Qs)

        # 反向傳播
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()


        # 訓練 actor 並更新 target network
        if self.train_steps % args.policy_frequency == 0:

            actor_actions = actor(states)

            actor_loss = -critic.forward_Q1(states, actor_actions).mean()

            # 反向傳播
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # 更新 target network
            with torch.no_grad():
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


