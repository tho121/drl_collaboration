from collections import namedtuple, deque
import torch
import numpy as np
import random
import copy


from agent import Agent, ReplayBuffer


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64       # minibatch size
GAMMA = 0.99            # discount factor

class MultiAgent():

    def __init__(self, num_agents, state_size, action_size):

        self.agents = []

        for i in range(num_agents):
            self.agents.append(Agent(state_size, action_size))

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

    def step(self, states, actions, rewards, next_states, done):

        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i],  done)

        if len(self.memory) > BATCH_SIZE:
            experiences, indexes = self.memory.sample()

            for agent in self.agents:
                error = agent.learn(experiences, GAMMA)
            
                #update priority replay memory
                self.memory.update(indexes, abs(error))

    def act(self, states, add_noise=True, noise_weight=1.0):
        
        actions = []

        for i in range(len(self.agents)):
            actions.append(self.agents[i].act(states[i], add_noise, noise_weight))

        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()

