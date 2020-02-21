from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from .config import config


class Drawer:
    logger = config.logger
    fontsize = 24
    font = ImageFont.truetype("arial.ttf", fontsize)

    def __init__(self):
        self.logger.debug('Drawer')
        self.logger.debug('Config at: ' + str(config.CONFIG_FILE))
        self.logger.debug(config.CONFIG)

    def draw_objects(self, image, objs, labels):
        draw = ImageDraw.Draw(image)
        for obj in objs:
            bbox = obj.bbox
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                           outline='blue')
            draw.text((bbox.xmin + 10, bbox.ymin + 10),
                      '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                      fill='blue', font=self.font)

    def draw_info(self, image, frame_meta):
        draw = ImageDraw.Draw(image)
        offset = 20
        for item in frame_meta:
            offset += 20
            draw.text((10, offset),
                      '%s\n' % (item + ': ' + str(frame_meta[item])),
                      fill='yellow', font=self.font)
