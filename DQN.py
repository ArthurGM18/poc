from Utils import preprocess, get_samples

import numpy as np
from time import time
from tqdm import trange
import tensorflow as tf

target_net_update_steps = 1000

# NN learning settings
batch_size = 64

# Training regime 
test_episodes_per_epoch = 5

num_train_epochs = 5
learning_steps_per_epoch = 8400

# Other parameters
frames_per_action = 10

class DQN(object):

    def __init__(self, actions):
        self.num_train_epochs = num_train_epochs
        self.learning_steps_per_epoch = learning_steps_per_epoch
        self.actions = actions

    def run(self, agent, game, replay_memory):
        time_start = time()

        for episode in range(self.num_train_epochs):
            train_scores = []
            print("\nEpoch %d\n-------" % (episode + 1))

            game.new_episode()

            for i in trange(self.learning_steps_per_epoch, leave=False):
                state = game.get_state()
                screen_buf = preprocess(state.screen_buffer)
                action = agent.choose_action(screen_buf)
                reward = game.make_action(self.actions[action], frames_per_action)
                done = game.is_episode_finished()

                if not done:
                    next_screen_buf = preprocess(game.get_state().screen_buffer)
                else:
                    next_screen_buf = tf.zeros(shape=screen_buf.shape)

                if done:
                    train_scores.append(game.get_total_reward())

                    game.new_episode()

                replay_memory.append((screen_buf, action, reward, next_screen_buf, done))

                if i >= batch_size:
                    agent.train_dqn(get_samples(replay_memory))
        
                if ((i % target_net_update_steps) == 0):
                    agent.update_target_net()

            train_scores = np.array(train_scores)
            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                    "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            self.test(test_episodes_per_epoch, game, agent)
            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))


    def test(self, test_episodes_per_epoch, game, agent):
        test_scores = []

        print("\nTesting...")
        for test_episode in trange(test_episodes_per_epoch, leave=False):
            game.new_episode()
            while not game.is_episode_finished():
                state = preprocess(game.get_state().screen_buffer)
                best_action_index = agent.choose_action(state)
                game.make_action(self.actions[best_action_index], frames_per_action)

            r = game.get_total_reward()
            test_scores.append(r)

        test_scores = np.array(test_scores)
        print("Results: mean: %.1f±%.1f," % (
            test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
            "max: %.1f" % test_scores.max())