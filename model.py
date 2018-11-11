import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):

    def __init__(self, state_size, action_size, unitsList):

        super(Actor, self).__init__()
        
        tempLayers = list()
        tempLayers.append(nn.Linear(state_size, unitsList[0]))

        lastIndex = len(unitsList)
        for i in range(1, lastIndex):
            tempLayers.append(nn.Linear(unitsList[i - 1], unitsList[i]))

        tempLayers.append(nn.Linear(unitsList[lastIndex - 1], action_size))

        self.layers = nn.ModuleList(tempLayers)
        self.reset_parameters()

    def reset_parameters(self):

        for i, l in enumerate(self.layers):
            l.weight.data.uniform_(*hidden_init(l))

        self.layers[len(self.layers) - 1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):

        x = F.relu(self.layers[0](state))

        for i in range(1, len(self.layers) - 1):
            x = F.relu(self.layers[i](x))

        return torch.tanh(self.layers[len(self.layers) - 1](x))


class Critic(nn.Module):

    def __init__(self, state_size, action_size, unitsList):

        super(Critic, self).__init__()

        tempLayers = list()
        tempLayers.append(nn.Linear(state_size, unitsList[0]))
        tempLayers.append(nn.Linear(unitsList[0] + action_size, unitsList[1]))

        lastIndex = len(unitsList)
        for i in range(2, lastIndex):
            tempLayers.append(nn.Linear(unitsList[i - 1], unitsList[i]))

        tempLayers.append(nn.Linear(unitsList[lastIndex - 1], 1))

        self.layers = nn.ModuleList(tempLayers)
        self.reset_parameters()

    def reset_parameters(self):
        
        for i, l in enumerate(self.layers):
            l.weight.data.uniform_(*hidden_init(l))

        self.layers[len(self.layers) - 1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):

        xs = F.relu(self.layers[0](state))
        x = torch.cat((xs, action), dim=1)

        for i in range(1, len(self.layers) - 1):
            x = F.relu(self.layers[i](x))

        return self.layers[len(self.layers) - 1](x)