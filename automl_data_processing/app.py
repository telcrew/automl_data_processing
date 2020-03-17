from . import video_read
from . import object_detection_inference
from . import draw
from . import display
from . import automl_prep

class Blueprint:

    @staticmethod
    def run():
        print("Hello World...")
        drawer = draw.Drawer()
        automlprep = automl_prep.AutoMLPrep(drawer)

        videoreader = video_read.VideoReader()
        inferenceworker = object_detection_inference.InferenceWorker(
            videoreader,
            automlprep,
        )
        display.Displayer(
            inferenceworker,
            drawer,
        )
