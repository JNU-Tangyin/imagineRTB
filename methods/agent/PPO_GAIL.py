import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions.categorical import Categorical

from methods.agent.network import Actor, Critic, Discriminator, RTB_ExpertTraj


class PPO_GAIL:
    def __init__(self, camp_id, state_dim, action_dim, action_value, hidden_dim,
                 lr, device, exper_traj_path, clip_param=0.2, max_grad_norm=0.5, ppo_update_time=10):
        super(PPO_GAIL, self).__init__()
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.discriminator = Discriminator(state_dim, action_value, hidden_dim).to(device)
        self.buffer = []
        self.expert = RTB_ExpertTraj(camp_id, exper_traj_path)
        self.loss_fn = nn.BCELoss()

        # ppo param
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.ppo_update_time = ppo_update_time

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.device = device

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32,  device=self.device)
        action_prob = self.actor(state)
        m = Categorical(action_prob)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob.item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic(state)
        return value.item()

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self, n_iter, batch_size):
        # ppo update
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float, device=self.device)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long, device=self.device).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float, device=self.device).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + 0.99 * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float, device=self.device)    
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), batch_size, False):
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor(state[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

        del self.buffer[:] # clear experience
        # GAIL update
        for i in range(n_iter):
            # sample expert transitions
            exp_state, exp_action = self.expert.sample(batch_size)
            exp_state = torch.tensor(exp_state, device=self.device, dtype=torch.float32)
            exp_action = torch.tensor(exp_action, device=self.device, dtype=torch.float32)
            
            # sample expert states for actor
            state, _ = self.expert.sample(batch_size)
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            action = Categorical(self.actor(state)).sample()
            
            # label tensors
            exp_label= torch.full((batch_size,1), 1.0, device=self.device)
            policy_label = torch.full((batch_size,1), 0.0, device=self.device)
            # with expert transitions
            prob_exp = self.discriminator(exp_state, exp_action)
            loss = self.loss_fn(prob_exp, exp_label)
            
            # update discriminator
            self.optim_discriminator.zero_grad()
            prob_policy = self.discriminator(state, action.detach())
            loss += self.loss_fn(prob_policy, policy_label)
            loss.backward()
            self.optim_discriminator.step()

            # update policy
            self.actor_optimizer.zero_grad()
            loss_actor = -self.discriminator(state, action)
            loss_actor.mean().backward()
            self.actor_optimizer.step()



