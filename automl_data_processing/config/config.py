from pathlib import Path
from typing import List, Tuple
import yaml
from loguru import logger
import os
from os.path import expanduser
from shutil import copyfile
import pkg_resources

VERSION = '0.0.1'
CONFIG_DIR = str(Path(__file__).parent)
CONFIG_FILE_NAME = 'config.yml'
CONFIG_FILE = Path(CONFIG_DIR) / Path(CONFIG_FILE_NAME)

logger.debug('Config at: ' + str(CONFIG_FILE))

GLOBAL_EXIT_SIGNAL = False

def load_yml(config_file):
    logger.debug('Loading config from ' + str(CONFIG_FILE))
    with open(config_file, 'r') as f:
        return yaml.full_load(f)

def load_config():
    if CONFIG_FILE.exists():
        return load_yml(CONFIG_FILE)
    else:
        logger.error('No config file ' + str(CONFIG_FILE))
        exit()

CONFIG = load_config()

 


