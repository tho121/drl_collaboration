# drl_collaboration
# Project 3: Collaboration with MADDPG by Tony Ho

### Project Environment Details

This project is about training two agents to pass a ball between each other without letting it touch the net in the middle or dropping to the ground.
By keeping the ball in play, the agents will recieve a reward of +0.1 for each time the ball goes to the other side of the net.
Hitting the net or the dropping the ball results in -0.01 reward for that agent.

For this project, the environment is considered solved when the last 100 episodes average a score of at least 0.5 for one of the agents

The observational space is of type continuous with a size of 8 with each agent receiving their own local observation. 
The action space is of type continuous with a size of 2 per agent

### Getting Started

Make sure to install the packages Unity ML-Agents, NumPy, PyTorch (v0.4) and Matlibplot  
Also, install Unity3D with the Linux Build Support option enabled

### Instructions

From the command line, type in 'python ./PROJECT_PATH/drl_continuous.py'
Running this file will start the training process again.
The environment is already included for Windows.  If another environment needs to be used, change the line

env = UnityEnvironment(file_name='./Tennis_Windows_x86_64/Tennis.exe')

to point to your environment.  This environment was downloaded from a link found in the lesson "3. The Environment - Explore" in the Collaboration and Competition topic.
