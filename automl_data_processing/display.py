from threading import Thread
import numpy as np
import cv2
from .config import config


class Displayer:
    display_width = config.CONFIG['display_width']
    display_height = config.CONFIG['display_height']
    inferenceworker = None
    drawer = None
    logger = config.logger

    def __init__(
        self,
        inferenceworker,
        drawer,
    ):
        self.logger.debug('Displayer')
        self.logger.debug('Config at: ' + str(config.CONFIG_FILE))
        self.logger.debug(config.CONFIG)
        self.inferenceworker = inferenceworker
        self.drawer = drawer

        displayer_thread = Thread(target=self.display_image)
        displayer_thread.start()
        self.logger.debug('displayer_thread started')

    def display_image(self,):
        while not config.GLOBAL_EXIT_SIGNAL:
            if self.inferenceworker.new_frame:
                image = self.inferenceworker.videoreader.capture_frame_rgb

                if image and self.inferenceworker.objs:
                    self.drawer.draw_objects(
                        image,
                        self.inferenceworker.objs,
                        self.inferenceworker.labels,
                    )

                if image:
                    self.drawer.draw_info(
                        image,
                        self.inferenceworker.frame_meta,
                    )
                    self.inferenceworker.new_frame = False
                    image_np = np.array(image)
                    image_np = image_np[:, :, ::-1].copy()

                    image_np = cv2.resize(image_np, (self.display_width, self.display_height))
                    cv2.imshow('processing', image_np)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        config.GLOBAL_EXIT_SIGNAL = True
                        break
