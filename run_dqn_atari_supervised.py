#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import argparse
import numpy as np
from deep_rl import *

# DQN
def dqn_pixel_atari(name, game):
    config = Config()
    config.sim_env = False
    config.history_length = 4
    log_dir = get_default_log_dir(dqn_pixel_atari.__name__)
    config.task_fn = lambda: Task(name, log_dir=log_dir)
    config.eval_env = Task(name, episode_life=False)

    config.optimizer_fn = lambda params: torch.optim.Adam(
        params, lr=np.sqrt(128 / 16) * 2.5e-4)
    config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    config.replay_fn = lambda: AsyncDummyReplay(game=game.lower(),
                                                batch_size=128,
                                                num_img_obs=config.history_length)

    config.batch_size = 128
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = int(1e4)
    config.exploration_steps = 0
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.double_q = False
    config.max_steps = int(2e7)
    config.log_interval = int(1e5)
    config.eval_interval = int(1e5)
    config.eval_episodes = 10
    config.logger = get_logger(tag='dqn_atari_supervised_' + game.lower())

    n_trials = 2
    for i in range(n_trials):
        exp_args = dict(
            exp='dqn_atari_superised',
            name=game,
            run_ID=i,
        )

        run_steps(DQNAgent(config), exp_args)

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
    dqn_pixel_atari(game, args.game.lower())
