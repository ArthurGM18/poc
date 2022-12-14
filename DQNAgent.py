from CNN import CNN
from Utils import extractDigits, split_tuple

import json
import platform
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

if platform.system() == 'Windows': 
    CONFIG_FILE = 'settings\\configs.json'
else:
    CONFIG_FILE = 'settings/configs.json'

with open(CONFIG_FILE, 'r') as f:
    data = json.load(f)


class DQNAgent:
    def __init__(self, num_actions):
        self.epsilon = data['DQN']['epsilon']
        self.epsilon_min = data['DQN']['epsilon_min']
        self.epsilon_decay = data['DQN']['epsilon_decay']
        self.discount_factor = data['DQN']['discount_factor']
        self.optimizer = SGD(data['DQN']['learning_rate'])
        self.num_actions = num_actions

        if data['model']['load']:
            self.dqn = tf.keras.models.load_model(data['model']['model_savefolder'])
        else:
            self.dqn = CNN(self.num_actions)
            self.target_net = CNN(self.num_actions)


    def update_target_net(self):
        self.target_net.set_weights(self.dqn.get_weights())


    def choose_action(self, state):
        if self.epsilon < np.random.uniform(0,1):
            action = int(tf.argmax(self.dqn(tf.reshape(state, (1,30,45,3))), axis=1))
        else:
            action = np.random.choice(range(self.num_actions), 1)[0]

        return action


    def train_dqn(self, samples):
        screen_buf, actions, rewards, next_screen_buf, dones = split_tuple(samples)

        row_ids = list(range(screen_buf.shape[0]))

        ids = extractDigits(row_ids, actions)
        done_ids = extractDigits(np.where(dones)[0])

        with tf.GradientTape() as tape:
            tape.watch(self.dqn.trainable_variables)

            Q_prev = tf.gather_nd(self.dqn(screen_buf), ids)
            
            Q_next = self.target_net(next_screen_buf)
            Q_next = tf.gather_nd(Q_next, extractDigits(row_ids, tf.argmax(self.dqn(next_screen_buf), axis=1)))
            
            q_target = rewards + self.discount_factor * Q_next

            if len(done_ids)>0:
                done_rewards = tf.gather_nd(rewards, done_ids)
                q_target = tf.tensor_scatter_nd_update(tensor=q_target, indices=done_ids, updates=done_rewards)

            td_error = tf.keras.losses.MSE(q_target, Q_prev)

        gradients = tape.gradient(td_error, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.dqn.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min