import argparse
import os
from os.path import join, exists
from itertools import count

import numpy as np
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Aggregate experience data')
parser.add_argument('--game', dest='game', default='breakout', type=str)
parser.add_argument('--start', dest='start', default=0, type=int)
parser.add_argument('--reward_threshold', dest='reward_threshold', type=int)
parser.add_argument('--ratio', dest='ratio',  type=float)
parser.add_argument('--mode', dest='mode', type=str)
args = parser.parse_args()

assert args.mode in ['reward', 'sample']

game, start = args.game, args.start
reward_threshold = args.reward_threshold
ratio = args.ratio
mode = args.mode

data_folder = join('data', 'atari_data', game)
states, actions, rewards, dones = [], [], [], []
for i in count(start=start):
    try:
        states.append(np.expand_dims(np.load(join(data_folder, f'{game}_states_{i}.npy')), axis=1))
        actions.append(np.load(join(data_folder, f'{game}_actions_{i}.npy')))
        rewards.append(np.load(join(data_folder, f'{game}_rewards_{i}.npy')))
        dones.append(np.load(join(data_folder, f'{game}_dones_{i}.npy')))
    except:
        break

print(f"Collected data ids from {start} (inclusive) to {i} (exclusive)")

states = np.concatenate(states, axis=0)
actions = np.concatenate(actions, axis=0)
rewards = np.concatenate(rewards, axis=0)
dones = np.concatenate(dones, axis=0)

if mode == 'reward':
    if reward_threshold is None:
        new_states = states
        new_actions = actions
        new_rewards = rewards
        new_dones = dones
    else:
        # Calculate episode index intervals
        done_idxs = np.argwhere(dones).reshape(-1)
        episode_intervals = [(0, done_idxs[0])]
        for i in range(len(done_idxs) - 1):
            episode_intervals.append((done_idxs[i]+1, done_idxs[i+1]))

        if done_idxs[-1] != len(dones) - 1:
            episode_intervals.append((done_idxs[-1], len(dones)-1))

        episode_rewards = []
        # Calculate episode rewards
        for start, end in episode_intervals:
            episode_rewards.append(np.sum(rewards[start:end]))
        episode_rewards = np.array(episode_rewards)

        rewards_idx = np.argsort(episode_rewards)
        rewards_sorted = episode_rewards[rewards_idx]
        rewards_idx = rewards_idx[rewards_sorted > reward_threshold]

        new_states, new_actions, new_rewards, new_dones = [], [], [], []
        for idx in rewards_idx:
            start, end = episode_intervals[idx]
            new_states.append(states[start:end])
            new_actions.append(actions[start:end])
            new_rewards.append(rewards[start:end])
            new_dones.append(dones[start:end])

        new_states = np.concatenate(new_states, axis=0)
        new_actions = np.concatenate(new_actions, axis=0)
        new_rewards = np.concatenate(new_rewards, axis=0)
        new_dones = np.concatenate(new_dones, axis=0)
elif mode == 'sample':
    assert ratio is not None and 0 < ratio <= 1
    # Calculate episode index intervals
    done_idxs = np.argwhere(dones).reshape(-1)
    episode_intervals = [(0, done_idxs[0])]
    for i in range(len(done_idxs) - 1):
        episode_intervals.append((done_idxs[i]+1, done_idxs[i+1]))

    if done_idxs[-1] != len(dones) - 1:
        episode_intervals.append((done_idxs[-1], len(dones)-1))

    episode_intervals = np.array(episode_intervals)
    n_episodes = len(episode_intervals)
    indices = np.random.choice(n_episodes, size=int(n_episodes * ratio), replace=False)
    episode_choices = episode_intervals[indices]
    
    new_states, new_actions, new_rewards, new_dones = [], [], [], []
    for start, end in episode_choices:
        new_states.append(states[start:end])
        new_actions.append(actions[start:end])
        new_rewards.append(rewards[start:end])
        new_dones.append(dones[start:end])

    new_states = np.concatenate(new_states, axis=0)
    new_actions = np.concatenate(new_actions, axis=0)
    new_rewards = np.concatenate(new_rewards, axis=0)
    new_dones = np.concatenate(new_dones, axis=0)

to_save_folder = join('data', 'atari_data', 'processed', game)
if not os.path.exists(to_save_folder):
    os.mkdir(to_save_folder)
np.save(join(to_save_folder, f'{game}_states.npy'), new_states)
np.save(join(to_save_folder, f'{game}_actions.npy'), new_actions)
np.save(join(to_save_folder, f'{game}_rewards.npy'), new_rewards)
np.save(join(to_save_folder, f'{game}_dones.npy'), new_dones)
print("Data Saved")

print(f"Data Shapes: {new_states.shape}, {new_actions.shape}, {new_rewards.shape}, {new_dones.shape}")
