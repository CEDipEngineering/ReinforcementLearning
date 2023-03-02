import random
from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from QLearning import QLearning
from numpy import loadtxt

env = gym.make("Taxi-v3", render_mode='ansi').env


EPISODES = 1500
paramList = [
    {
        "alpha": 0.1,
        "gamma": 0.6,
        "epsilon": 0.7,
        "epsilon_min": 0.05,
        "epsilon_dec": 0.99,
        "episodes": EPISODES
    },
    {
        "alpha": 0.45,
        "gamma": 0.6,
        "epsilon": 0.7,
        "epsilon_min": 0.05,
        "epsilon_dec": 0.99,
        "episodes": EPISODES
    },
    {
        "alpha": 0.99,
        "gamma": 0.6,
        "epsilon": 0.7,
        "epsilon_min": 0.05,
        "epsilon_dec": 0.99,
        "episodes": EPISODES
    },
    {
        "alpha": 0.1,
        "gamma": 0.1,
        "epsilon": 0.7,
        "epsilon_min": 0.05,
        "epsilon_dec": 0.99,
        "episodes": EPISODES
    },
    {
        "alpha": 0.1,
        "gamma": 0.5,
        "epsilon": 0.7,
        "epsilon_min": 0.05,
        "epsilon_dec": 0.99,
        "episodes": EPISODES
    },
    {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.7,
        "epsilon_min": 0.05,
        "epsilon_dec": 0.5,
        "episodes": EPISODES
    },
    {
        "alpha": 0.1,
        "gamma": 0.6,
        "epsilon": 0.99,
        "epsilon_min": 0.05,
        "epsilon_dec": 0.99,
        "episodes": EPISODES
    },
    {
        "alpha": 0.1,
        "gamma": 0.6,
        "epsilon": 0.5,
        "epsilon_min": 0.05,
        "epsilon_dec": 0.99,
        "episodes": EPISODES
    },
    {
        "alpha": 0.1,
        "gamma": 0.6,
        "epsilon": 0.1,
        "epsilon_min": 0.05,
        "epsilon_dec": 0.99,
        "episodes": EPISODES
    },
]

for i,params in enumerate(paramList):
    # only execute the following lines if you want to create a new q-table
    qlearn = QLearning(env, **params)
    q_table = qlearn.train('data/q-table-taxi-driver-{}.csv'.format(i), 'results/actions_taxidriver', actionsName='data/action-history-{}.csv'.format(i))
    #q_table = loadtxt('data/q-table-taxi-driver.csv', delimiter=',')
exit(0)

(state, _) = env.reset()
epochs, penalties, reward = 0, 0, 0
done = False
frames = [] # for animation
    
while not done:
    print(state)
    action = np.argmax(q_table[state])
    state, reward, done, truncated, info = env.step(action)

    if reward == -10:
        penalties += 1

    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(),
        'state': state,
        'action': action,
        'reward': reward
        }
    )
    epochs += 1

from IPython.display import clear_output
from time import sleep

clear_output(wait=True)

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        #print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)

print("\n")
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))