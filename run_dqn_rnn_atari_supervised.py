#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import argparse
import numpy as np
from deep_rl import *

# DQN RNN
def dqn_rnn_pixel_atari(name, game):
    config = Config()
    config.sim_env = False
    config.seq_len = 10
    config.warmup = 5
    log_dir = get_default_log_dir(dqn_rnn_pixel_atari.__name__)
    config.task_fn = lambda: Task(name, log_dir=log_dir, stack_size=config.seq_len)
    config.eval_env = Task(name, episode_life=False, stack_size=1)

    config.optimizer_fn = lambda params: torch.optim.RMSprop(
        params, lr=np.sqrt(128 / 16) * 2.5e-4, alpha=0.95, eps=0.01, centered=True)
    config.network_fn = lambda: VanillaNet(config.action_dim, LSTMConvBody(warmup=config.warmup,
                                                                           seq_len=config.seq_len,
                                                                           in_channels=1))
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    config.replay_fn = lambda: DummySeqReplay(game=game.lower(),
                                              seq_len=config.seq_len,
                                              warmup=config.warmup,
                                              batch_size=128)

    config.batch_size = 128
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = int(8e4)
    config.exploration_steps = 0
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.double_q = False
    config.max_steps = int(6e5)
    config.log_interval = int(2e3)
    config.eval_interval = int(1e4)
    config.eval_episodes = 1
    config.logger = get_logger(tag='dqn_lstm_atari_supervised_' + game.lower())
    run_steps(DQNAgent(config))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DQN with supervised data')
    parser.add_argument('--game', dest='game', default='Breakout', type=str)
    parser.add_argument('--device', dest='device', default=0, type=int)

    args = parser.parse_args()

    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    select_device(args.device)

    game = '{}NoFrameskip-v4'.format(args.game)
    dqn_rnn_pixel_atari(game, args.game.lower())
