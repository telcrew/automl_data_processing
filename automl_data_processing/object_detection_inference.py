from threading import Thread
from PIL import Image
import os
from pathlib import Path
from os.path import expanduser
from datetime import datetime
import statistics

from .config import config
from .utils import detect
import tflite_runtime.interpreter as tflite


class InferenceWorker:

    EDGETPU_SHARED_LIB = "libedgetpu.so.1.0"
    model_path = config.CONFIG['model_path']
    model_file = config.CONFIG['model_file']
    labels_file = config.CONFIG['labels_file']
    labels_file = str(Path(__file__).parent) + str(Path(model_path) / Path(labels_file))
    detect_objects = config.CONFIG['detect_objects']
    confidence = config.CONFIG['confidence']
    detection_sampling_rate = int(config.CONFIG['detection_sampling_rate'])
    target_samples = int(config.CONFIG['target_samples'])
    model = str(Path(__file__).parent) + str(Path(model_path) / Path(model_file))
    logger = config.logger

    videoreader = None
    interpreter = None
    automlprep = None

    labels = None
    objs = None
    detections = None
    frame_meta = None
    new_frame = None

    detections_history = []
    detections_history_output_path = config.CONFIG['detections_history_output_path']

    avg_confidence = []

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

    def initialise_engine(self,):
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

    def object_detection_inference(self,):
        self.interpreter, self.labels = self.initialise_engine()

        sample_count = 0
        detection_count = 0
        average_confidence = 0

        while not config.GLOBAL_EXIT_SIGNAL:
            if not self.new_frame:
                time_stamp1 = datetime.now()
                self.detections = []
                self.frame_meta = {}

                capture_frame_rgb = self.videoreader.capture_frame_rgb
                currentvideo = self.videoreader.currentvideo
                frame_number = self.videoreader.frame_number

                if capture_frame_rgb is not None:
                    scale = detect.set_input(
                        self.interpreter,
                        capture_frame_rgb.size,
                        lambda size: capture_frame_rgb.resize(
                            size,
                            Image.ANTIALIAS))
                    self.interpreter.invoke()
                    try:
                        self.objs = detect.get_output(self.interpreter, self.confidence, scale)
                        print(self.objs)
                    except:
                        self.objs = None
                        self.logger.debug('detect.get_output fail')
                        pass

                    num_detect = 0
                    inference_time = -1

                    if self.objs:
                        for obj in self.objs:
                            if (obj.score >= self.confidence
                                    and (self.labels[obj.id] in self.detect_objects
                                         or "all" in self.detect_objects)):

                                num_detect += 1

                                detection = {}
                                detection['currentvideo'] = currentvideo
                                detection['frame_number'] = frame_number
                                detection['resolution'] = capture_frame_rgb.size
                                detection['label_id'] = obj.id
                                detection['label'] = self.labels[obj.id]
                                detection['score'] = obj.score
                                detection['box'] = obj.bbox
                                detection['numdetect'] = num_detect
                                self.avg_confidence.append(obj.score)
                                average_confidence = round(statistics.mean(self.avg_confidence),2)

                                self.detections.append(detection)
                                self.detections_history.append(detection)

                        time_stamp2 = datetime.now()
                        inference_time = (time_stamp2 - time_stamp1).microseconds
                        self.logger.debug(self.detections)

                    self.frame_meta['currentvideo'] = currentvideo
                    self.frame_meta['model'] = self.model_file
                    self.frame_meta['labels'] = self.labels
                    self.frame_meta['confidence'] = str(self.confidence) + ' (' + str(average_confidence) + ')'
                    self.frame_meta['frame_number'] = frame_number
                    self.frame_meta['resolution'] = capture_frame_rgb.size
                    self.frame_meta['numdetect'] = num_detect
                    self.frame_meta['inference_time'] = inference_time

                    if frame_number == 0:
                        detection_count = 0
                    if len(self.detections) > 0 and self.target_samples > 0:
                        detection_count += 1
                        if detection_count >= self.detection_sampling_rate:
                            sample_count += 1
                            detection_count = 0
                            self.automlprep.automl_save_frame(capture_frame_rgb,
                                currentvideo + '_' +
                                str(frame_number) + '_' + str(len(self.detections)) + detection['label'],
                                self.detections,
                                self.frame_meta,
                                self.labels
                                )
                        elif sample_count >= self.target_samples:
                            config.GLOBAL_EXIT_SIGNAL = True
                    
                    if config.SAVE_FRAME == True:
                        self.automlprep.automl_save_frame(capture_frame_rgb,
                                currentvideo + '_' +
                                str(frame_number),
                                {},
                                {},
                                {},
                                )
                        config.SAVE_FRAME = False

                    self.new_frame = True

        if self.target_samples > 0:
            self.automlprep.automl_save_csv()

        self.detections_dump()


    def detections_dump(self,):
        if self.detections_history:
            content = ''
            try:
                self.detections_history_output_path = expanduser(self.detections_history_output_path)
            except BaseException:
                pass
            
            try:
                Path(self.detections_history_output_path).mkdir(parents=True, exist_ok=True)
            except:
                raise
        
            filepath = Path(self.detections_history_output_path) / Path(self.model_file + '_detections_history_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.json')
            
            for detection in self.detections_history:
                content = content + str(detection) + "\n"

            with open(filepath, 'a') as f:
                f.write(content)

            self.logger.debug('detections history at: ' + str(filepath))
            self.logger.debug('content: ' + content)


    
