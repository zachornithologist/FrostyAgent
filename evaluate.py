import gym
import numpy as np
from agents.frosty_agent import FrostyAgent

def evaluate_agent():
    env = gym.make('FrozenLake-v1', is_slippery=False)
    agent = FrostyAgent(env.observation_space.n, env.action_space.n, 0, 0, 0)
    agent.load('logs/q_table.npy')
    
    num_test_episodes = 100
    successes = 0

    for episode in range(num_test_episodes):
        state = env.reset()
        done = False
        
        while not done:
                        action = np.argmax(agent.q_table[state, :])
            new_state, reward, done, _ = env.step(action)
            state = new_state
            
            if done and reward == 1.0:
                successes += 1

    success_rate = successes / num_test_episodes * 100
    print(f"Success rate: {success_rate}%")

if __name__ == "__main__":
    evaluate_agent()
            