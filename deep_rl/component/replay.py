#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import sys
from os.path import join
from collections import deque

import numpy as np

import torch
import torch.multiprocessing as mp

from ..utils import *

DATA_PATH = join('data', 'atari_data', 'processed')

class DummySeqReplay(object):

    def __init__(self, game, batch_size, seq_len, warmup,
                 clip_reward=True):
        self.episode_intervals, self.data = self.load_episodes(join(DATA_PATH, game),
                                                               game)
        self.seq_len = seq_len
        self.warmup = warmup
        self.clip_reward = clip_reward
        self.batch_size = batch_size * (seq_len - warmup)
        self.true_batch_size = batch_size

        valid_seq_idx_ranges = list()
        for start, end in self.episode_intervals:
            if end - start + 1 < self.seq_len:
                continue
            valid_seq_idx_ranges.append((0, end - self.seq_len + 2))
        self.valid_seq_idx_ranges = valid_seq_idx_ranges

        total = sum([end - start + 1 for start, end in self.valid_seq_idx_ranges])
        self.valid_seq_idx_weights = [(end - start + 1) / total
                                      for start, end in self.valid_seq_idx_ranges]

    def load_data(self, data_path, game):
        states = np.load(join(data_path, "{}_states.npy".format(game)))
        actions = np.load(join(data_path, "{}_actions.npy".format(game)))
        rewards = np.load(join(data_path, "{}_rewards.npy".format(game)))
        dones = np.load(join(data_path, "{}_dones.npy".format(game)))

        dones[-1] = True

        return states, actions, rewards, dones

    def load_episodes(self, data_path, game):
        states, actions, rewards, dones = self.load_data(data_path, game)
        done_idxs = np.argwhere(dones).reshape(-1)
        episode_intervals = [(0, done_idxs[0])]
        for i in range(len(done_idxs) - 1):
            episode_intervals.append((done_idxs[i]+1, done_idxs[i+1]))

        if done_idxs[-1] != len(dones) - 1:
            episode_intervals.append((done_idxs[-1], len(dones)-1))
        return episode_intervals, (states, actions, rewards, dones)

    def sample(self):
        batch_size = self.true_batch_size
        states, actions, rewards, dones = self.data
        valid_seq_idx_ranges = self.valid_seq_idx_ranges
        valid_seq_idx_weights = self.valid_seq_idx_weights

        range_idx = np.random.choice(np.arange(len(valid_seq_idx_ranges)), size=batch_size,
                                     p=valid_seq_idx_weights)
        range_idx = [valid_seq_idx_ranges[i] for i in range_idx]
        batch_idx = [np.random.randint(start, end + 1)
                     for start, end in range_idx]

        state_batch = np.concatenate([states[i:i + self.seq_len]
                                      for i in batch_idx])
        action_batch = np.concatenate([actions[i + self.warmup:i + self.seq_len]
                                       for i in batch_idx])
        reward_batch = np.concatenate([rewards[i + self.warmup:i + self.seq_len]
                                       for i in batch_idx])
        if self.clip_reward:
            reward_batch = np.sign(reward_batch)

        dones_batch = np.concatenate([dones[i:i + self.seq_len]
                                      for i in batch_idx])

        next_idxs = np.array([idx + 1 if not dones[idx] else idx for idx in batch_idx])
        next_state_batch = np.concatenate([states[i:i + self.seq_len]
                                           for i in next_idxs])
        dones_batch = np.concatenate([dones[i+self.warmup:i + self.seq_len]
                                      for i in batch_idx])

        return state_batch, action_batch, reward_batch, next_state_batch, dones_batch


        def feed_batch(self, experience):
            pass

        def size(self):
            return sys.maxsize

class AsyncDummyReplay(mp.Process):
    SAMPLE = 0
    EXIT = 1

    def __init__(self, game, batch_size,
                 clip_reward=True, include_time_dim=False,
                 num_img_obs=4):
        mp.Process.__init__(self)
        self.pipe, self.worker_pipe = mp.Pipe()
        self.batch_size = batch_size
        self.cache_len = 2
        self.game = game
        self.clip_reward = clip_reward
        self.include_time_dim = include_time_dim
        self.num_img_obs = num_img_obs

        self.start()

    def run(self):
        torch.cuda.is_available()
        replay = DummyReplay(self.game, self.batch_size,
                             clip_reward=self.clip_reward,
                             include_time_dim=self.include_time_dim,
                             num_img_obs=self.num_img_obs)
        cache = []
        first = True
        cur_cache = 0

        def set_up_cache():
            batch_data = replay.sample()
            batch_data = [tensor(x) for x in batch_data]
            for i in range(self.cache_len):
                cache.append([x.clone() for x in batch_data])
                for x in cache[i]: x.share_memory_()
            sample(0)
            sample(1)

        def sample(cur_cache):
            batch_data = replay.sample()
            batch_data = [tensor(x) for x in batch_data]
            for cache_x, x in zip(cache[cur_cache], batch_data):
                cache_x.copy_(x)

        while True:
            op, data = self.worker_pipe.recv()
            if op == self.SAMPLE:
                if first:
                    set_up_cache()
                    first = False
                    self.worker_pipe.send([cur_cache, cache])
                else:
                    self.worker_pipe.send([cur_cache, None])
                cur_cache = (cur_cache + 1) % 2
                sample(cur_cache)
            elif op == self.EXIT:
                self.worker_pipe.close()
                return
            else:
                raise Exception('Unknown command')

    def feed(self, exp):
        pass

    def feed_batch(self, exps):
        pass

    def sample(self):
        self.pipe.send([self.SAMPLE, None])
        cache_id, data = self.pipe.recv()
        if data is not None:
            self.cache = data
        return self.cache[cache_id]

    def close(self):
        self.pipe.send([self.EXIT, None])
        self.pipe.close()

class DummyReplay(object):

    def __init__(self, game, batch_size,
                 clip_reward=True, include_time_dim=False,
                 num_img_obs=4):
        self.data = self.load_data(join(DATA_PATH, game), game)

        states, _, _, dones = self.data
        frame_valid = np.zeros(len(states), dtype='int64')
        for i in range(len(states)):
            for j in range(1, num_img_obs):
                if i - j < 0 or dones[i - j]:
                    break
            frame_valid[i] = j
        self.frame_valid = frame_valid
        self.clip_reward = clip_reward
        self.include_time_dim = include_time_dim
        self.num_img_obs = num_img_obs
        self.batch_size = batch_size

    def load_data(self, data_path, game):
        states = np.load(join(data_path, "{}_states.npy".format(game)))
        actions = np.load(join(data_path, "{}_actions.npy".format(game)))
        rewards = np.load(join(data_path, "{}_rewards.npy".format(game)))
        dones = np.load(join(data_path, "{}_dones.npy".format(game)))

        dones[-1] = True

        return states, actions, rewards, dones

    def sample(self):
        batch_size = self.batch_size
        states, actions, rewards, dones = self.data
        frame_valid = self.frame_valid

        batch_idx = np.random.randint(0, len(states), size=batch_size)

        state_blanks = frame_valid[batch_idx]
        state_batch = np.concatenate([states[np.maximum(batch_idx - i, batch_idx - state_blanks + 1)]
                                      for i in range(self.num_img_obs - 1, -1, -1)], axis=1).astype('float32')

        if self.include_time_dim:
            action_batch = np.concatenate([actions[idx - self.num_img_obs + 1 : idx + 1]
                                           for idx in batch_idx]).astype('float32')
            reward_batch = np.concatenate([rewards[idx - self.num_img_obs + 1 : idx + 1]
                                           for idx in batch_idx]).astype('float32')
            dones_batch = np.concatenate([dones[idx - self.num_img_obs + 1 : idx + 1]
                                          for idx in batch_idx]).astype('float32')
        else:
            action_batch = actions[batch_idx]
            reward_batch = rewards[batch_idx]
            dones_batch = dones[batch_idx]

        if self.clip_reward:
            reward_batch = np.sign(reward_batch)

        next_idxs = np.array([idx + 1 if not dones[idx] else idx for idx in batch_idx])
        next_state_blanks = frame_valid[next_idxs]
        next_state_batch = np.concatenate([states[np.maximum(next_idxs - i, next_idxs - next_state_blanks + 1)]
                                           for i in range(self.num_img_obs - 1, -1, -1)], axis=1).astype('float32')

        if self.include_time_dim:
            state_batch = state_batch.reshape((batch_size * self.num_img_obs, 1,
                                               state_batch.shape[2], state_batch.shape[3]))
            next_state_batch = next_state_batch.reshape((batch_size * self.num_img_obs, 1,
                                                         next_state_batch.shape[2],
                                                         next_state_batch.shape[3]))

        return state_batch, action_batch, reward_batch, next_state_batch, dones_batch

    def feed_batch(self, experience):
        pass

    def size(self):
        return sys.maxsize

class Replay:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0

    def feed(self, experience):
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if self.empty():
            return None
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        return batch_data

    def size(self):
        return len(self.data)

    def empty(self):
        return not len(self.data)


class SkewedReplay:
    def __init__(self, memory_size, batch_size, criterion):
        self.replay1 = Replay(memory_size // 2, batch_size // 2)
        self.replay2 = Replay(memory_size // 2, batch_size // 2)
        self.criterion = criterion

    def feed(self, experience):
        if self.criterion(experience):
            self.replay1.feed(experience)
        else:
            self.replay2.feed(experience)

    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    def sample(self):
        data1 = self.replay1.sample()
        data2 = self.replay2.sample()
        if data2 is not None:
            data = list(map(lambda x: np.concatenate(x, axis=0), zip(data1, data2)))
        else:
            data = data1
        return data


class AsyncReplay(mp.Process):
    FEED = 0
    SAMPLE = 1
    EXIT = 2
    FEED_BATCH = 3

    def __init__(self, memory_size, batch_size):
        mp.Process.__init__(self)
        self.pipe, self.worker_pipe = mp.Pipe()
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.cache_len = 2
        self.start()

    def run(self):
        torch.cuda.is_available()
        replay = Replay(self.memory_size, self.batch_size)
        cache = []
        pending_batch = None

        first = True
        cur_cache = 0

        def set_up_cache():
            batch_data = replay.sample()
            batch_data = [tensor(x) for x in batch_data]
            for i in range(self.cache_len):
                cache.append([x.clone() for x in batch_data])
                for x in cache[i]: x.share_memory_()
            sample(0)
            sample(1)

        def sample(cur_cache):
            batch_data = replay.sample()
            batch_data = [tensor(x) for x in batch_data]
            for cache_x, x in zip(cache[cur_cache], batch_data):
                cache_x.copy_(x)

        while True:
            op, data = self.worker_pipe.recv()
            if op == self.FEED:
                replay.feed(data)
            elif op == self.FEED_BATCH:
                if not first:
                    pending_batch = data
                else:
                    for transition in data:
                        replay.feed(transition)
            elif op == self.SAMPLE:
                if first:
                    set_up_cache()
                    first = False
                    self.worker_pipe.send([cur_cache, cache])
                else:
                    self.worker_pipe.send([cur_cache, None])
                cur_cache = (cur_cache + 1) % 2
                sample(cur_cache)
                if pending_batch is not None:
                    for transition in pending_batch:
                        replay.feed(transition)
                    pending_batch = None
            elif op == self.EXIT:
                self.worker_pipe.close()
                return
            else:
                raise Exception('Unknown command')

    def feed(self, exp):
        self.pipe.send([self.FEED, exp])

    def feed_batch(self, exps):
        self.pipe.send([self.FEED_BATCH, exps])

    def sample(self):
        self.pipe.send([self.SAMPLE, None])
        cache_id, data = self.pipe.recv()
        if data is not None:
            self.cache = data
        return self.cache[cache_id]

    def close(self):
        self.pipe.send([self.EXIT, None])
        self.pipe.close()


class Storage:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'm',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean']
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            assert k in self.keys
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)
