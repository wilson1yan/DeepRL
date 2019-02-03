import argparse
import os
from os.path import join, exists
from itertools import count

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Aggregate experience data')
parser.add_argument('--game', dest='game', default='breakout', type=str)
parser.add_argument('--start', dest='start', default=0, type=int)
parser.add_argument('--process', dest='process', action='store_true')
args = parser.parse_args()

game, start, process = args.game, args.start, args.process

if process:
    data_folder = join('data', 'atari_data', game)
    states, actions, rewards, dones = [], [], [], []
    for i in count(start):
        try:
            states.append(np.load(join(data_folder, f'{game}_states_{i}.npy')))
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

    to_save_folder = join('data', 'atari_data', 'processed', game)
    if not os.path.exists(to_save_folder):
        os.mkdir(to_save_folder)
    np.save(join(to_save_folder, f'{game}_states.npy'), states)
    np.save(join(to_save_folder, f'{game}_actions.npy'), actions)
    np.save(join(to_save_folder, f'{game}_rewards.npy'), rewards)
    np.save(join(to_save_folder, f'{game}_dones.npy'), dones)
    print("Data Saved")
else:
    to_save_folder = join('data', 'atari_data', 'processed', game)
    np.load(join(to_save_folder, f'{game}_states.npy'))
    np.load(join(to_save_folder, f'{game}_actions.npy'))
    np.load(join(to_save_folder, f'{game}_rewards.npy'))
    np.load(join(to_save_folder, f'{game}_dones.npy'))
    print("Data Loaded")

print("Performing Reward Analysis...")
print(f"Data Shapes: {states.shape}, {actions.shape}, {rewards.shape}, {dones.shape}")

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

plt.figure()
plt.hist(episode_rewards)
plt.title(f'{game} rewards')
plt.save_fig('rewards.png')
