import os
import vizdoom as vzd

config_file_path = 'scenarios/MAPA02.cfg'


class Doom(object):

    def __init__(self):
        pass

    def initialize_game(self):
        print("Initializing doom...")
        game = vzd.DoomGame()
        game.load_config(config_file_path)
        game.set_window_visible(False)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        game.init()
        print("Doom initialized.")

        return game

    