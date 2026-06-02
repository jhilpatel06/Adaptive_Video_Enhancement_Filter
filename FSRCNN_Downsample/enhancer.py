import cv2
from cv2 import dnn_superres


class FaceEnhancer:
    def __init__(self, model_path="FSRCNN_x4.pb", scale=4):
        self.sr = dnn_superres.DnnSuperResImpl_create()
        self.scale = scale
        try:
            self.sr.readModel(model_path)
            self.sr.setModel("fsrcnn", scale)
            self.loaded = True
            print("FSRCNN model loaded successfully.")
        except Exception as exc:
            print(f"Error loading FSRCNN model: {exc}")
            self.loaded = False

    def enhance(self, frame):
        if not self.loaded or frame.size == 0:
            return frame
        return self.sr.upsample(frame)
