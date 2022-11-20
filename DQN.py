from Utils import preprocess, get_samples

import json
import platform
import numpy as np
from time import time
from tqdm import trange
import tensorflow as tf

if platform.system() == 'Windows': 
    CONFIG_FILE = 'settings\\configs.json'
else:
    CONFIG_FILE = 'settings/configs.json'

with open(CONFIG_FILE, 'r') as f:
    data = json.load(f)


class DQN(object):

    def __init__(self, actions):
        self.num_train_epochs = data['training']['num_train_epochs']
        self.learning_steps_per_epoch = data['training']['learning_steps_per_epoch']
        self.actions = actions

    def run(self, agent, game, replay_memory):
        
        time_start = time()
        train_file = open(f"output/{data['training']['num_train_epochs']}_{data['training']['learning_steps_per_epoch']}_train.txt", 'w')
        test_file = open(f"output/{data['training']['num_train_epochs']}_{data['training']['learning_steps_per_epoch']}_test.txt", 'w')

        for episode in range(self.num_train_epochs):
            train_scores = []


            print("\nEpoch %d\n-------" % (episode + 1))

            game.new_episode()

            for i in trange(self.learning_steps_per_epoch, leave=False):
                state = game.get_state()
                screen_buf = preprocess(state.screen_buffer)
                action = agent.choose_action(screen_buf)
                reward = game.make_action(self.actions[action], data['DoomGame']['frames_per_action'])
                done = game.is_episode_finished()

                if not done:
                    next_screen_buf = preprocess(game.get_state().screen_buffer)
                else:
                    next_screen_buf = tf.zeros(shape=screen_buf.shape)

                if done:
                    train_scores.append(game.get_total_reward())

                    game.new_episode()

                replay_memory.append((screen_buf, action, reward, next_screen_buf, done))

                if i >= 64:
                    agent.train_dqn(get_samples(replay_memory))
        
                if ((i % data['DQN']['target_net_update_steps']) == 0):
                    agent.update_target_net()

            train_scores = np.array(train_scores)
            train_file.write(
                "%.1f, %.1f, %.1f, %.1f, %.2f\n" % (
                    train_scores.mean(), 
                    train_scores.std(), 
                    train_scores.min(), 
                    train_scores.max(),
                    (time() - time_start) / 60.0
                )
            )

            test_scores = self.test(data['training']['test_episodes_per_epoch'], game, agent)

            test_file.write(
                "%.1f, %.1f, %.1f, %.1f\n" % (
                    test_scores.mean(), 
                    test_scores.std(), 
                    test_scores.min(), 
                    test_scores.max()
                )
            )


        train_file.close()
        test_file.close()


    def test(self, test_episodes_per_epoch, game, agent):
        test_scores = []

        print("\nTesting...")
        for _ in trange(test_episodes_per_epoch, leave=False):
            game.new_episode()
            while not game.is_episode_finished():
                state = preprocess(game.get_state().screen_buffer)
                best_action_index = agent.choose_action(state)
                game.make_action(self.actions[best_action_index], data['DoomGame']['frames_per_action'])

            r = game.get_total_reward()
            test_scores.append(r)

        return np.array(test_scores)