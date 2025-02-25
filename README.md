# Kolmogrov-Arnold-Networks-based-RL-Performance-Gym

## Overview

This project implements a Kolmogrov Arnold Networks based Deep Q-Network (DQN) to train an agent to solve the OpenAI Gym environment LunarLander-v2. The training process utilizes reinforcement learning techniques with an epsilon-greedy policy, experience replay, and a target network.

## Dependencies

Make sure you have the following dependencies installed:

`pip install -r requirements.txt`


## Training Process

Initialize Environment and Agent

The environment is set up using OpenAI Gym (LunarLander-v2).
The DQNAgent can be switched between KAN and MLP implementation. 

## Saving the Model and Scores

The trained model weights are saved as a .pth file.

Training scores are saved as .npy files.

##Running the Code

For KAN the code uses FASTKAN library
To train the agent, run:

`python train.py`

This will train the agent and save the model weights and scores.


## Changing Model Architecture

You can train different models (kan or mlp) by modifying:

model = 'kan'  # or 'mlp'
hidden_size = 54  # Change hidden layer size
agent = DQNAgent(state_size=state_size, action_size=action_size, hidden_size=hidden_size, model=model)

## Visualizing Training Results

The script includes a function plot_scores() to visualize training performance:

plot_scores(scores, f'dqn_agent_{model}_{hidden_size}')

This will generate a PNG file with the training progress.

Saving and Loading the Model

To save the trained model:

torch.save(agent.q_network.state_dict(), 'dqn_agent_kan_54.pth')

To load the model later:

agent.q_network.load_state_dict(torch.load('dqn_agent_kan_54.pth'))

## License

This project is open-source under the MIT License.
