from threading import Thread
from PIL import Image
import os
from pathlib import Path
from datetime import datetime

from .config import config
from .utils import detect
import tflite_runtime.interpreter as tflite


class InferenceWorker:

    EDGETPU_SHARED_LIB = "libedgetpu.so.1.0"
    model_path = config.CONFIG['model_path']
    model = config.CONFIG['model']
    labels_file = config.CONFIG['labels_file']
    labels_file = str(Path(__file__).parent) + str(Path(model_path) / Path(labels_file))
    detect_objects = config.CONFIG['detect_objects']
    confidence = config.CONFIG['confidence']
    model = str(Path(__file__).parent) + str(Path(model_path) / Path(model))
    logger = config.logger

    videoreader = None
    interpreter = None
    automlprep = None

    labels = None
    objs = None
    detections = None
    frame_meta = None
    new_frame = None

    def __init__(
        self,
        videoreader,
        automlprep,
    ):

        self.logger.debug('InferenceWorker')
        self.logger.debug('Config at: ' + str(config.CONFIG_FILE))
        self.logger.debug(config.CONFIG)

        self.videoreader = videoreader
        self.automlprep = automlprep
        object_detection_inference_thread = Thread(target=self.object_detection_inference)
        object_detection_inference_thread.start()
        self.logger.debug('object_detection_inference_thread started')

    def read_label_file(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
        return ret

    def initialise_engine(self):
        try:
            model_file, *device = self.model.split('@')
            interpreter = tflite.Interpreter(
                model_path=model_file,
                experimental_delegates=[
                    tflite.load_delegate(self.EDGETPU_SHARED_LIB,
                                         {'device': device[0]} if device else {})
                ])

            labels = self.read_label_file(self.labels_file)
            interpreter.allocate_tensors()

        except RuntimeError as re:
            self.logger.error(str(re))
        except ValueError as ve:
            self.logger.error(str(ve))
            self.logger.error('Is your coral pluggen in?!')
            config.GLOBAL_EXIT_SIGNAL = True
            exit()

        return interpreter, labels

    def object_detection_inference(self):
        self.interpreter, self.labels = self.initialise_engine()

        while not config.GLOBAL_EXIT_SIGNAL:
            if not self.new_frame:
                time_stamp1 = datetime.now()
                self.detections = []
                self.frame_meta = {}

                if self.videoreader.capture_frame_rgb is not None:
                    scale = detect.set_input(
                        self.interpreter,
                        self.videoreader.capture_frame_rgb.size,
                        lambda size: self.videoreader.capture_frame_rgb.resize(
                            size,
                            Image.ANTIALIAS))
                    self.interpreter.invoke()
                    self.objs = detect.get_output(self.interpreter, self.confidence, scale)

                    num_detect = 0
                    inference_time = -1

                    if self.objs:
                        for obj in self.objs:
                            if (obj.score >= self.confidence
                                    and (self.labels[obj.id] in self.detect_objects
                                         or "all" in self.detect_objects)):

                                num_detect += 1

                                # xmin, ymin, xmax, ymax = obj.bbox

                                detection = {}
                                detection['label_id'] = str(obj.id)
                                detection['label'] = self.labels[obj.id]
                                detection['score'] = str(obj.score)
                                detection['box'] = obj.bbox
                                detection['numdetect'] = num_detect

                                self.detections.append(detection)

                                self.automlprep.automl_save_frame(self.videoreader.capture_frame_rgb,
                                                      self.videoreader.currentvideo + '_' +
                                                      str(self.videoreader.frame_number) + '.jpg',
                                                      self.detections,
                                                      self.frame_meta,
                                                      )

                        time_stamp2 = datetime.now()
                        inference_time = (time_stamp2 - time_stamp1).microseconds
                        self.logger.debug(self.detections)

                    self.frame_meta['currentvideo'] = self.videoreader.currentvideo
                    self.frame_meta['frame_number'] = self.videoreader.frame_number
                    self.frame_meta['resolution'] = self.videoreader.capture_frame_rgb.size
                    self.frame_meta['numdetect'] = num_detect
                    self.frame_meta['inference_time'] = inference_time
                    self.logger.debug(self.frame_meta)

                    

                    self.new_frame = True
        
        self.automlprep.automl_save_csv()
