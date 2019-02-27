#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import argparse
from deep_rl import *

def dqn_pixel_atari(name, game):
    config = Config()
    config.history_length = 4
    log_dir = get_default_log_dir(dqn_pixel_atari.__name__)
    config.game = game
    config.sim_env = True
    config.task_fn = lambda: Task(name, log_dir=log_dir)
    config.eval_env = Task(name, episode_life=False)

    config.optimizer_fn = lambda params: torch.optim.RMSprop(
        params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    # config.network_fn = lambda: DuelingNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    # config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e6), batch_size=32)

    config.batch_size = 32
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.store_data = True
    # config.double_q = True
    config.double_q = False
    config.max_steps = int(2e7)
    config.eval_interval = int(1e5)
    config.logger = get_logger(tag=dqn_pixel_atari.__name__)
    run_steps(DQNAgent(config))


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

    game = '{}NoFrameskip-v4'.format(args.game)
    dqn_pixel_atari(game, args.game.lower())
