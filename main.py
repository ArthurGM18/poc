from DQNAgent import DQNAgent
from Utils import preprocess
from DoomGame import Doom
from DQN import DQN

import vizdoom as vzd
from time import sleep
import itertools as it
import tensorflow as tf
from collections import deque

tf.executing_eagerly()
tf.compat.v1.enable_eager_execution()


replay_memory_size = 10000

frames_per_action = 10
episodes_to_watch = 5

save_model = True
load = False
skip_learning = False
watch = True

model_savefolder = "./model"

if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print("GPU available")
    DEVICE = "/gpu:0"
else:
    print("No GPU available")
    DEVICE = "/cpu:0"


if __name__ == '__main__':
    agent = DQNAgent()
    game = Doom().initialize_game()
    replay_memory = deque(maxlen=replay_memory_size)

    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    
    with tf.device(DEVICE):

        if not skip_learning:
            print("Starting the training!")

            DQN(actions).run(agent, game, replay_memory)

            game.close()
            print("======================================")
            print("Training is finished.")

            if save_model:
                agent.dqn.save(model_savefolder)

        game.close()

        if watch:
            game.set_window_visible(True)
            game.set_mode(vzd.Mode.ASYNC_PLAYER)
            game.init()

            for _ in range(episodes_to_watch):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    best_action_index = agent.choose_action(state)

                    # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                    game.set_action(actions[best_action_index])
                    for _ in range(frames_per_action):
                        game.advance_action()

                # Sleep between episodes
                sleep(1.0)
                score = game.get_total_reward()
                print("Total score: ", score)
