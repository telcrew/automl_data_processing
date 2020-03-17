from pathlib import Path
import os
from os.path import expanduser
from config import config

detections_history_output_path = config.CONFIG['detections_history_output_path']
try:
    detections_history_output_path = expanduser(detections_history_output_path)
except BaseException:
    pass

def plot():
    files = os.listdir(detections_history_output_path)
    content = ''
    for file in files:
        if file.endswith(".json"):
            filepath = Path(detections_history_output_path) / Path(file)
            with open(filepath, 'r') as f:
                content = content + f.read()

    print(content)

plot()