from threading import Thread
from os.path import expanduser
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from .config import config


class VideoReader:

    footage_path = config.CONFIG['footage_path']
    footage_files = config.CONFIG['footage_files']
    width = config.CONFIG['width']
    height = config.CONFIG['height']
    device_video = bool(config.CONFIG['device_video'])

    currentvideo = None
    default_image_np_global = np.zeros([width, height, 3], dtype=np.uint8)
    capture_frame_rgb_np = None
    capture_frame_rgb = None

    logger = config.logger
    frame_number = 0

    def __init__(self):
        self.logger.debug('VideoRead')
        self.logger.debug('Config at: ' + str(config.CONFIG_FILE))
        self.logger.debug(config.CONFIG)
        video_read_thread = Thread(target=self.video_read)
        video_read_thread.start()
        self.logger.debug('video_read_thread started')

    def video_read(self):
        for footage_file in self.footage_files:
            self.logger.debug('Processing: ' + footage_file)
            self.currentvideo = footage_file
            frame_number = 0

            if self.device_video:
                video = 0
            else:
                video = Path(self.footage_path + '/' + footage_file)
                try:
                    video = expanduser(Path(self.footage_path + '/' + footage_file))
                except BaseException:
                    pass

            cap = cv2.VideoCapture(video)

            while(cap.isOpened() and not config.GLOBAL_EXIT_SIGNAL):
                ret, frame = cap.read()
                try:
                    if frame is not None:
                        self.frame_number += 1
                        frame = cv2.resize(frame, (self.width, self.height))
                        ar = frame
                        ar = ar[:, :, 0:3]
                        (im_height, im_width, _) = frame.shape
                        self.default_image_np_global = np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)

                        self.capture_frame_rgb_np = cv2.cvtColor(self.default_image_np_global, cv2.COLOR_BGR2RGB)
                        self.capture_frame_rgb = Image.fromarray(self.capture_frame_rgb_np)
                    else:
                        break

                except RuntimeError as re:
                    self.logger.debug("default_capture_thread:" + str(re))

                # cv2.imshow('video_read', self.default_image_np_global)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

            cap.release()
            cv2.destroyAllWindows()

        config.GLOBAL_EXIT_SIGNAL = True
