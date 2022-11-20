from DQNAgent import DQNAgent
from Utils import preprocess
from DoomGame import Doom
from DQN import DQN

import vizdoom as vzd
import tensorflow as tf
from collections import deque

import os
import json
import platform
from time import sleep
import itertools as it


if platform.system() == 'Windows': 
    CONFIG_FILE = 'settings\\configs.json'
else:
    CONFIG_FILE = 'settings/configs.json'

with open(CONFIG_FILE, 'r') as f:
    data = json.load(f)

if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print("GPU available")
    DEVICE = "/gpu:0"
else:
    print("No GPU available")
    DEVICE = "/cpu:0"

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["TF_NUM_INTRAOP_THREADS"] = "6"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_soft_device_placement(True)

if __name__ == '__main__':


    # -------------------------------------------------------------------------
    # BLOCO DE INICIALIZACOES
    # -------------------------------------------------------------------------
    
    game = Doom().initialize_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # FIM DA INICIALIZACAO DO AMBIENTE VIRTUAL
    # -------------------------------------------------------------------------

    agent = DQNAgent(num_actions=n)

    # FIM DA INICIALIZACAO DO AGENTE
    # -------------------------------------------------------------------------

    # Inicializacao da memoria de Replay
    replay_memory = deque(maxlen=data['DQN']['replay_memory_size'])


    # FIM DO BLOCO DE INICIALIZACAO
    # -------------------------------------------------------------------------


    with tf.device(DEVICE):

        if not data['model']['skip_learning']:


            # -----------------------------------------------------------------
            # BLOCO DE TREINAMENTO
            # -----------------------------------------------------------------
            
            # -----------------------------------------------------------------
            # INICIA TREINAMENTO

            DQN(actions).run(agent, game, replay_memory)
            game.close()

            # FIM DO TREINAMENTO
            # -----------------------------------------------------------------

            if data['model']['save_model']:
                agent.dqn.save(data['model']['model_savefolder'])

            # SALVA MODELO TREINADO
            # -----------------------------------------------------------------

            # FIM DO BLOCO DE TREINAMENTO
            # -----------------------------------------------------------------


        game.close()

        # ---------------------------------------------------------------------
        # BLOCO DE TESTE
        # ---------------------------------------------------------------------

        if data['model']['watch']:
            game.set_window_visible(True)
            game.set_mode(vzd.Mode.ASYNC_PLAYER)
            game.init()

            for _ in range(data['DoomGame']['episodes_to_watch']):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    best_action_index = agent.choose_action(state)

                    game.set_action(actions[best_action_index])
                    for _ in range(data['DoomGame']['frames_per_action']):
                        game.advance_action()

                sleep(1.0)
                score = game.get_total_reward()
                print("Total score: ", score)
        
        # FIM DO BLOCO DE TESTE
        # ---------------------------------------------------------------------
