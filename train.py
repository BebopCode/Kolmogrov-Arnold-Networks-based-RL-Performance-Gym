import gym 
from DQNAgent import *
import matplotlib.pyplot as plt
import random

def initialize_env_settings(env):
    # Ensure the environment is reset to initialize its state
    env.reset()

    # Randomize the initial state
    # These ranges are approximations and may need adjustment
    env.lander.position = np.array([
        np.random.uniform(-0.2, 0.2),  # x position
        np.random.uniform(0.8, 1.2)    # y position
    ])
    env.lander.linearVelocity = np.array([
        np.random.uniform(-0.1, 0.1),  # x velocity
        np.random.uniform(-0.1, 0.1)   # y velocity
    ])
    env.lander.angle = np.random.uniform(-0.2, 0.2)  # angle
    env.lander.angularVelocity = np.random.uniform(-0.1, 0.1)  # angular velocity

    # Randomize leg positions
    for i in range(2):
        env.legs[i].ground_contact = False
        env.legs[i].position = env.lander.position + np.array([0.2 * (-1 if i == 0 else 1), -0.2])


def plot_scores(episode_durations,str):
    plt.figure(figsize=(10, 5))
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    
    plt.title('Training Result')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(durations_t.numpy(), label='Episode duration')
    
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='100-episode average')
    
    plt.legend()
    plt.savefig(f'{str}.png')
    plt.close()  # Close the figure to free up memory



def train(agent, env, n_episodes=800, eps_start=1.0, eps_end=0.01, eps_decay=0.995, target_update=10):
    '''
    Train a DQN agent.
    
    Parameters
    ----------
    agent: DQNAgent
        The agent to be trained.
    env: gym.Env
        The environment in which the agent is trained.
    n_episodes: int, default=2000
        The number of episodes for which to train the agent.
    eps_start: float, default=1.0
        The starting epsilon for epsilon-greedy action selection.
    eps_end: float, default=0.01
        The minimum value that epsilon can reach.
    eps_decay: float, default=0.995
        The decay rate for epsilon after each episode.
    target_update: int, default=10
        The frequency (number of episodes) with which the target network should be updated.
        
    Returns
    -------
    list of float
        The total reward obtained in each episode.
    '''

    # Initialize the scores list and scores window
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    # Loop over episodes
    for i_episode in range(1, n_episodes + 1):
        
        # Reset environment and score at the start of each episode
        initialize_env_settings(env)
        state, _ = env.reset()
        score = 0 

        # Loop over steps
        while True:
            
            # Select an action using current agent policy then apply in environment
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action) 
            done = terminated or truncated
            
            # Update the agent, state and score
            agent.step(state, action, reward, next_state, done)
            state = next_state 
            score += reward

            # End the episode if done
            if done:
                break 
        
        # At the end of episode append and save scores
        scores_window.append(score)
        scores.append(score) 

        # Decrease epsilon
        eps = max(eps_end, eps_decay * eps)

        # Print some info
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}", end="")

        # Update target network every target_update episodes
        if i_episode % target_update == 0:
            agent.update_target_network()
            
        # Print average score every 100 episodes
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            
        

    return scores


# Make an environment
env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = 'kan'
hidden_size = 54
agent = DQNAgent(state_size=state_size, action_size=action_size,hidden_size= hidden_size, model=model, learning_rate=1e-4)
print(f'Model:{model}, hidden size: {hidden_size}')
# Train it

scores = train(agent, env)
plot_scores(scores,f'dqn_agent_{model}_{hidden_size}')
torch.save(agent.q_network.state_dict(), f'dqn_agent_{model}_{hidden_size}.pth')
np.save(f'scores_{model}_{hidden_size}.npy', np.array(scores))
print("Agent saved successfully.")
'''
# Initilize a DQN agent
for i in range(1,10):
    model = 'kan'
    hidden_size = i*5
    agent = DQNAgent(state_size=state_size, action_size=action_size,hidden_size= hidden_size, model=model)
    print(f'Model:{model}, hidden size: {hidden_size}')
    # Train it
    scores = train(agent, env)
    torch.save(agent.q_network.state_dict(), f'dqn_agent_{model}_{hidden_size}.pth')
    print("Agent saved successfully.")

for i in range(1,10):
    model = 'mlp'
    hidden_size = i*10
    agent = DQNAgent(state_size=state_size, action_size=action_size,hidden_size= hidden_size,model=model)
    print(f'Model:{model}, hidden size: {hidden_size}')
    # Train it
    scores = train(agent, env)
    torch.save(agent.q_network.state_dict(), f'dqn_agent_{model}_{hidden_size}.pth')
    print("Agent saved successfully.")
'''