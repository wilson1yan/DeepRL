#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from tqdm import tqdm
import pickle
import os
import datetime
import torch
import time
from .torch_utils import *
try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path

def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    pbar = None
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag))
        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards):
            if config.sim_env:
                rewards = agent.episode_rewards
                agent.episode_rewards = []
                config.logger.info('total steps %d, returns %.2f/%.2f/%.2f/%.2f (mean/median/min/max), %.2f steps/s' % (
                    agent.total_steps, np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards),
                    config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            if not config.sim_env:
                if pbar:
                    pbar.close()
            agent.eval_episodes()
            if not config.sim_env:
                pbar = tqdm(total=config.eval_interval)
        if not config.sim_env:
            pbar.update()
            pbar.set_description(desc='total steps %d' % (agent.total_steps))
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()

def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

def get_default_log_dir(name):
    return './log/%s-%s' % (name, get_time_str())

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()

def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]
