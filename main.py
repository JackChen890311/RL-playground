import random
import numpy as np
import pickle as pk
import gymnasium as gym  

def visualize_random():
    env = gym.make("Taxi-v3", render_mode="human")
    env.reset()
    random_number = lambda:random.randint(0,5)

    print("Action Space {}".format(env.action_space)) 
    print("State Space {}".format(env.observation_space))

    while True:
        result = env.step(random_number())
        print(result)
        env.render()

def train(env, episode = 100000):
    """Training the agent"""
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(episode):
        state, info = env.reset()
        epochs, penalties, reward, = 0, 0, 0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values
            next_state, reward, terminated, truncated, info = env.step(action) 
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1
            state = next_state
            epochs += 1
            
        if i % 10000 == 0:
            print(f"Episode: {i}")
    
    with open('q_table.pk','wb') as f:
        pk.dump(q_table,f)

    print("Training finished.")
    return q_table

def test(env, episodes, q_table):
    """Evaluate agent's performance after Q-learning"""
    total_epochs, total_penalties, total_reward = 0, 0, 0
    
    for _ in range(episodes):
        state, info = env.reset()
        env.render()
        epochs, penalties, reward_epoch = 0, 0, 0
        
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            if reward == -10:
                penalties += 1
            epochs += 1
            reward_epoch += reward
    
        total_penalties += penalties
        total_epochs += epochs
        total_reward += reward_epoch
    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average rewards per episode: {total_reward / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")


if __name__ == '__main__':
    # env = gym.make("Taxi-v3")
    # _, _ = env.reset()
    # q_table = train(env, 100000)


    env = gym.make("Taxi-v3", render_mode = 'human')
    _, _ = env.reset()
    with open('q_table.pk','rb') as f:
        q_table = pk.load(f)
    test(env, 10, q_table)