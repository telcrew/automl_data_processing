from threading import Thread
import numpy as np
import cv2
from PIL import Image
from .config import config


class Displayer:
    width = config.CONFIG['width']
    height = config.CONFIG['height']
    display_width = config.CONFIG['display_width']
    display_height = config.CONFIG['display_height']
    display_annotate = bool(config.CONFIG['display_annotate'])
    display_info = bool(config.CONFIG['display_info'])
    inferenceworker = None
    drawer = None

    logger = config.logger

    object_zoom_on = bool(config.CONFIG['object_zoom_on'])
    zoom_delay = config.CONFIG['zoom_delay']
    zoom_count = 0
    zoom_roi = {}
    zoom_margin = config.CONFIG['zoom_margin']

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
                detections = self.inferenceworker.detections

                if image and self.inferenceworker.objs:
                    if self.display_annotate:
                        self.drawer.draw_objects(
                            image,
                            self.inferenceworker.objs,
                            self.inferenceworker.labels,
                        )

                if image:
                    frame_meta = self.inferenceworker.frame_meta
                    self.inferenceworker.new_frame = False
                    if self.display_info:
                        self.drawer.draw_info(
                            image,
                            frame_meta,
                        )
                    image_np = np.array(image)
                    image_np = image_np[:, :, ::-1].copy()

                    subimage_np = self.object_zoom(detections, image_np)
                    subimage_np = cv2.resize(subimage_np, (self.display_width, self.display_height))

                    cv2.imshow('processing', subimage_np)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        config.GLOBAL_EXIT_SIGNAL = True
                        break

    def object_zoom(self, detections, image_np):
        if self.object_zoom_on:
            if detections:
                self.zoom_count = 0
                margin = self.zoom_margin
                xmin, ymin, xmax, ymax = detections[0]['box']

                ar = self.width / self.height
                h = int(ymax - ymin)
                w = int(xmax - xmin)
                if h >= w:
                    w = h * ar
                else:
                    h = w / ar

                xmin_ = max(int(xmin - margin * w), 0)
                xmax_ = min(int(xmax + margin * w), self.width)

                ymin_ = max(int(ymin - margin * h), 0)
                ymax_ = min(int(ymax + margin * w), self.height)

                self.zoom_roi = {'xmin_': xmin_, 'ymin_': ymin_, 'xmax_': xmax_, 'ymax_': ymax_}
                sub_image_np = image_np[self.zoom_roi['ymin_']:self.zoom_roi['ymax_'],
                                        self.zoom_roi['xmin_']:self.zoom_roi['xmax_']]

                return sub_image_np
            elif self.zoom_count <= self.zoom_delay and self.zoom_roi:
                self.zoom_count += 1
                sub_image_np = image_np[self.zoom_roi['ymin_']:self.zoom_roi['ymax_'],
                                        self.zoom_roi['xmin_']:self.zoom_roi['xmax_']]
                return sub_image_np
            else:
                self.zoom_count = 0
                self.zoom_roi = {}
                return image_np
        else:
            return image_np
