import gym
import numpy as np
from config import config
from agents.frosty_agent import FrostyAgent

def train_agent():
    env = gym.make('FrozenLake-v1', is_slippery=False)
    agent = FrostyAgent(env.observation_space.n, env.action_space.n, config['learning_rate'], config['discount_factor'], config['epsilon'])

    for episode in range(config['num_episodes']):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, new_state)
            state = new_state

    agent.save('logs/q_table.npy')
    print("Training complete and Q-table saved!")

if __name__ == "__main__":
    train_agent()