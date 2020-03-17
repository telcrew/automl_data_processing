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
    dataset_purpose = 'TRAIN'
    detect_objects_prefix = config.CONFIG['detect_objects_prefix']
    create_annotated_images = bool(config.CONFIG['create_annotated_images'])
    target_samples = int(config.CONFIG['target_samples'])

    drawer = None
    now = None

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

        self.now = datetime.now()
        self.image_output_path = self.image_output_path + '/' + self.now.strftime("%Y%m%d%H%M%S")

        if self.target_samples > 0:
            Path(self.image_output_path).mkdir(parents=True, exist_ok=True)
            Path(self.image_output_path + '/annotated').mkdir(parents=True, exist_ok=True)

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
        if detections:
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
                    str(y_relative_max) + ',,' + \
                    "\n"
                self.csv_file_data = self.csv_file_data + file_line
                self.logger.debug(self.csv_file_data)
        else:
            file_line = self.dataset_purpose + "," + self.google_bucket + filename + ',' + \
                    '' + ',' + \
                    '' + ',' + \
                    '' + ',' + \
                    ',,' + \
                    '' + ',' + \
                    '' + ',,' + \
                    "\n"
            self.csv_file_data = self.csv_file_data + file_line
            self.logger.debug(self.csv_file_data)

    def automl_save_frame(
        self,
        frame,
        filename,
        detections,
        frame_meta,
        labels,
    ):

        Path(self.image_output_path).mkdir(parents=True, exist_ok=True)
        Path(self.image_output_path + '/annotated').mkdir(parents=True, exist_ok=True)
        filepath = Path(self.image_output_path) / Path(filename)
        frame.save(str(filepath) + '.jpg')

        if self.create_annotated_images and self.target_samples > 0:
            annotated_image = frame
            filepath = Path(self.image_output_path + '/annotated') / Path(filename)
            self.drawer.draw_objects(annotated_image, labels, detections)
            self.drawer.draw_info(annotated_image, frame_meta)
            annotated_image.save(str(filepath) + '_annotated.jpg')

        self.automl_data_csv(str(filename) + '.jpg', detections, frame_meta)

    def automl_save_csv(
        self,
    ):
        Path(self.image_output_path).mkdir(parents=True, exist_ok=True)
        Path(self.image_output_path + '/annotated').mkdir(parents=True, exist_ok=True)
        filepath = Path(self.image_output_path) / Path('annotations_' + self.now.strftime("%Y%m%d%H%M%S") + '.csv')

        f = open(filepath, "w")
        f.write(self.csv_file_data)
        f.close()

        self.logger.debug('annotations at ' + str(filepath))
