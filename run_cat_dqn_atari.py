#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import argparse
from deep_rl import *

# C51
def categorical_dqn_pixel_atari(name, game):
    config = Config()
    log_dir = get_default_log_dir(categorical_dqn_pixel_atari.__name__)
    config.game = game
    config.task_fn = lambda: Task(name, log_dir=log_dir)
    config.eval_env = Task(name, episode_life=False)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00025, eps=0.01 / 32)
    config.network_fn = lambda: CategoricalNet(config.action_dim, config.categorical_n_atoms, NatureConvBody())
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    # config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e6), batch_size=32)

    config.discount = 0.99
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.categorical_v_max = 10
    config.categorical_v_min = -10
    config.categorical_n_atoms = 51
    config.sgd_update_frequency = 4
    config.gradient_clip = 0.5
    config.max_steps = int(2e7)
    config.logger = get_logger(tag=categorical_dqn_pixel_atari.__name__ + '_' + game)
    run_steps(CategoricalDQNAgent(config))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Categorical DQN experiment to collect data')
    parser.add_argument('--game', dest='game', default='Breakout', type=str)
    parser.add_argument('--device', dest='device', default=0, type=int)
    
    args = parser.parse_args()

    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    select_device(args.device)

    game = '{}NoFrameskip-v0'.format(args.game)
    categorical_dqn_pixel_atari(game, args.game.lower())
