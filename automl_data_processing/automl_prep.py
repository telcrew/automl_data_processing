from pathlib import Path
import os
from os.path import expanduser
from PIL import Image
from datetime import datetime
from .config import config


class AutoMLPrep:
    logger = config.logger
    image_output_path = config.CONFIG['image_output_path']
    csv_file_data = ''
    google_bucket = config.CONFIG['google_bucket']
    dataset_purpose = config.CONFIG['dataset_purpose']
    detect_objects_prefix = config.CONFIG['detect_objects_prefix']

    drawer = None

    def __init__(
        self,
        drawer
    ):
        self.drawer = drawer
        self.logger.debug('AutoMLPrep')
        self.logger.debug('Config at: ' + str(config.CONFIG_FILE))
        self.logger.debug(config.CONFIG)

        try:
            self.image_output_path = expanduser(self.image_output_path)
        except BaseException:
            pass

        now = datetime.now()
        self.image_output_path = self.image_output_path + '/' + now.strftime("%Y%m%d%H%M%S")
        Path(self.image_output_path).mkdir(parents=True, exist_ok=True)

        test = os.listdir(self.image_output_path)
        for item in test:
            if item.endswith(".jpg"):
                os.remove(os.path.join(self.image_output_path, item))

    def automl_data_csv(
        self,
        filename,
        detections,
        frame_meta,
    ):
        for detection in detections:
            xmin, ymin, xmax, ymax = detection['box']
            x_relative_min = round(xmin / config.CONFIG['width'], 2)
            y_relative_min = round(ymin / config.CONFIG['height'], 2)
            x_relative_max = round(xmax / config.CONFIG['width'], 2)
            y_relative_max = round(ymax / config.CONFIG['height'], 2)

            file_line = self.dataset_purpose + "," + self.google_bucket + filename + ',' + \
                self.detect_objects_prefix + detection['label'] + ',' + \
                str(x_relative_min) + ',' + \
                str(y_relative_min) + ',' + \
                ',,' + \
                str(x_relative_max) + ',' + \
                str(y_relative_max) + ',' + \
                ',,' + \
                "\n"
            self.csv_file_data = self.csv_file_data + file_line
            self.logger.debug(self.csv_file_data)

    def automl_save_frame(
        self,
        frame,
        filename,
        detections,
        frame_meta,
        objs,
        labels,
    ):

        Path(self.image_output_path).mkdir(parents=True, exist_ok=True)
        filepath = Path(self.image_output_path) / Path(filename)
        frame.save(str(filepath) + '.jpg')

        annotated_image = frame
        self.drawer.draw_objects(annotated_image, objs, labels)
        annotated_image.save(str(filepath) + '_annotated.jpg')

        self.automl_data_csv(str(filename) + '.jpg', detections, frame_meta)

    def automl_save_csv(
        self,
    ):
        Path(self.image_output_path).mkdir(parents=True, exist_ok=True)
        filepath = Path(self.image_output_path) / Path('annotations.csv')

        f = open(filepath, "w")
        f.write(self.csv_file_data)
        f.close()

        self.logger.debug('annotations at ' + str(filepath))


# TRAIN,gs://cloud-ml-data/img/openimage/3/2520/3916261642_0a504acd60_o.jpg,Salad,0.0,0.0954,,,0.977,0.957,,
# VALIDATE,gs://cloud-ml-data/img/openimage/3/2520/3916261642_0a504acd60_o.jpg,Seafood,0.0154,0.1538,,,1.0,0.802,,
# TEST,gs://cloud-ml-data/img/openimage/3/2520/3916261642_0a504acd60_o.jpg,Tomato,0.0,0.655,,,0.231,0.839,,
# (x_relative_min, y_relative_min,,,x_relative_max,y_relative_max,,)
