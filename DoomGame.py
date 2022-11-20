import vizdoom as vzd

import json
import platform

if platform.system() == 'Windows': 
    CONFIG_FILE = 'settings\\configs.json'
else:
    CONFIG_FILE = 'settings/configs.json'

with open(CONFIG_FILE, 'r') as f:
    data = json.load(f)


class Doom(object):

    def __init__(self):
        pass

    def initialize_game(self):
        print("Initializing doom...")
        game = vzd.DoomGame()
        game.load_config(f"{data['DoomGame']['config_file_path']}/{data['DoomGame']['map_name']}")
        ##game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.init()
        print("Doom initialized.")

        return game

    