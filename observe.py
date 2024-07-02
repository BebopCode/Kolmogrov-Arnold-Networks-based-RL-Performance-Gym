import gym
env = gym.make('LunarLander-v2', render_mode="human")
from DQNAgent import *
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
import random
seed = random.randint(1, 10000)
# Initilize a DQN agent
model = 'mlp'
hidden_size= 128
agent = DQNAgent(state_size=state_size, action_size=action_size,model=model,hidden_size=hidden_size)
agent.q_network.load_state_dict(torch.load(f'dqn_agent_{model}_{hidden_size}.pth'))
agent.target_network.load_state_dict(torch.load(f'dqn_agent_{model}_{hidden_size}.pth'))
print("Agent loaded successfully.")
def play_DQN_episode(env, agent, seed):
    score = 0
    state, _ = env.reset(seed=seed)
    total_steps = 0
    while True:
        # eps=0 for predictions
        action = agent.act(state, 0)
        state, reward, terminated, truncated, _ = env.step(action) 
        done = terminated or truncated

        score += reward
        total_steps += 1
        # End the episode if done
        if done:
            break 
    print(total_steps)
    return score,total_steps
 
def run_multiple_episodes(env, agent, num_episodes=10):
    total_score = 0
    total_steps = 0
    for i in range(num_episodes):
        score,steps = play_DQN_episode(env, agent, seed=i)  # Use different seed for each episode
        total_score += score
        total_steps += steps
    
    average_score = total_score / num_episodes
    average_steps = total_steps/ num_episodes
    return average_score, average_steps

# Run 10 episodes and calculate the average score
average_score, average_steps = run_multiple_episodes(env, agent, num_episodes=10)
print(f"Average score over 10 episodes: {average_score:.2f}, avg steps: {average_steps}")