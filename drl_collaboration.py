from unityagents import UnityEnvironment

import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from multiagent import MultiAgent

env = UnityEnvironment(file_name='./Tennis_Windows_x86_64/Tennis.exe')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
 
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)

agents = MultiAgent(len(env_info.agents), state_size=state_size, action_size=action_size)

def ddpg(continuing=False, n_episodes=1000, max_t=3000, print_every=100):

    if continuing:
        for i in range(len(agents.agents)):
            agents.agents[i].actor_local.load_state_dict(torch.load('checkpoint_actor' + str(i) + '.pth'))
            agents.agents[i].critic_local.load_state_dict(torch.load('checkpoint_critic' + str(i) + '.pth'))

    scores_deque = deque(maxlen=print_every)
    all_scores = []

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]  
        states = env_info.vector_observations
        agents.reset()

        scores = []

        for i in range(len(agents.agents)):
            scores.append(0)

        noise_weight = max(0.001, 1.0 - (i_episode/(n_episodes * 0.9)))

        for t in range(max_t):

            actions = agents.act(states, add_noise=True, noise_weight=noise_weight)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            done = env_info.local_done[0]
            agents.step(states, actions, rewards, next_states, done)
            states = next_states

            for i in range(len(agents.agents)):
                scores[i] += rewards[i]

            if done:
                break

        scores_deque.append(max(scores))
        all_scores.append(max(scores))

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
            for i in range(len(agents.agents)):
                torch.save(agents.agents[i].actor_local.state_dict(), 'checkpoint_actor' + str(i) + '.pth')
                torch.save(agents.agents[i].critic_local.state_dict(), 'checkpoint_critic' + str(i) + '.pth')

            if np.mean(scores_deque) >= 0.5:
                print("Environment Solved at episode " + str(i_episode))
            
    return all_scores

def test(n_episodes=300, max_t=1000, print_every=100):

    for i in range(len(agents.agents)):
        agents.agents[i].actor_local.load_state_dict(torch.load('checkpoint_actor' + str(i) + '.pth'))
        agents.agents[i].critic_local.load_state_dict(torch.load('checkpoint_critic' + str(i) + '.pth'))

    scores_deque = deque(maxlen=print_every)
    all_scores = []

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]  
        states = env_info.vector_observations
        agents.reset()

        scores = []

        for i in range(len(agents.agents)):
            scores.append(0)

        for t in range(max_t):
            actions = agents.act(states)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            done = env_info.local_done[0]

            states = next_states
            for i in range(len(agents.agents)):
                scores[i] += rewards[i]

            if done:
                break

        scores_deque.append(max(scores))
        all_scores.append(max(scores))

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
            
    return all_scores



#scores = ddpg(continuing=False, n_episodes=3000)
scores = test(n_episodes=3)


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('figure1.png')
plt.show()



print("done")

