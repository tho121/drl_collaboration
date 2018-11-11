import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from priorityreplay import Memory

import torch
import torch.nn.functional as F
import torch.optim as optim


TAU = 1e-2            # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001        # L2 weight decay

device = torch.device("cpu") ##torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        #actor network (w/ target network)
        self.actor_local = Actor(state_size, action_size, (400, 300)).to(device)
        self.actor_target = Actor(state_size, action_size, (400, 300)).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        #critic network (w/ target network)
        self.critic_local = Critic(state_size, action_size, (400, 300)).to(device)
        self.critic_target = Critic(state_size, action_size, (400, 300)).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.noise = OUNoise(action_size)

        #self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

    '''
    def step(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > BATCH_SIZE:
            experiences, indexes = self.memory.sample()
            error = self.learn(experiences, GAMMA)
            
            #update priority replay memory
            self.memory.update(indexes, abs(error))
    '''
    
    def act(self, state, add_noise=True, noise_weight=1.0):

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()

        if add_noise:
            action += self.noise.sample() * noise_weight

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        #compute critic loss
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        td_error = critic_loss.data.cpu().numpy()

        #minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #update actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        #minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #update target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        return td_error

    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

class OUNoise:

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2, dt=1e-2, x0=None):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma * np.ones(size)
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        #self.state = copy.copy(self.mu)
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def sample(self):
        #x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        #self.state = x + dx
        #return self.state
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size):

        self.action_size = action_size
        #self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.priorityMemory = Memory(buffer_size)

    def add(self, state, action, reward, next_state, done):
        
        e = self.experience(state, action, reward, next_state, done)
        #self.memory.append(e)

        self.priorityMemory.store(e)

    def sample(self):

        #experiences = random.sample(self.memory, k=self.batch_size)

        indexes, experiences, _ = self.priorityMemory.sample(self.batch_size)

        states = torch.from_numpy(np.vstack([e[0].state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[0].action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e[0].reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[0].next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[0].done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones), indexes

    def update(self, indexes, absolute_error):
        self.priorityMemory.batch_update(indexes, absolute_error)

    def __len__(self):

        return self.priorityMemory.tree.tree.size