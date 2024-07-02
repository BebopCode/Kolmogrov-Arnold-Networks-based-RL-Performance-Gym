import gym
env = gym.make('LunarLander-v2', render_mode="human")
from DQNAgent import *
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
import random
seed = random.randint(1, 10000)
# Initilize a DQN agent
model = 'kan'
hidden_size= 40
agent = DQNAgent(state_size=state_size, action_size=action_size,model='kan',hidden_size=40)
agent.q_network.load_state_dict(torch.load(f'dqn_agent_{model}_{hidden_size}.pth'))
agent.target_network.load_state_dict(torch.load(f'dqn_agent_{model}_{hidden_size}.pth'))
print("Agent loaded successfully.")
def play_DQN_episode(env, agent):
    score = 0
    state, _ = env.reset(seed=seed)
    
    while True:
        # eps=0 for predictions
        action = agent.act(state, 0)
        state, reward, terminated, truncated, _ = env.step(action) 
        done = terminated or truncated

        score += reward

        # End the episode if done
        if done:
            break 

    return score
 
score = play_DQN_episode(env, agent)
print("Score obtained:", score)