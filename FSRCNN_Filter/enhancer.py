import cv2
from cv2 import dnn_superres

class FaceEnhancer:
    def __init__(self, model_path="FSRCNN_x3.pb", scale=3):
        """
        Initializes the FSRCNN Super Resolution model.
        """
        self.sr = dnn_superres.DnnSuperResImpl_create()
        self.scale = scale
        try:
            self.sr.readModel(model_path)
            self.sr.setModel("fsrcnn", scale)
            self.loaded = True
            print("FSRCNN model loaded successfully.")
        except Exception as e:
            print(f"Error loading FSRCNN model: {e}")
            self.loaded = False

    def enhance(self, roi):
        """
        Upscales the Region of Interest using FSRCNN.
        """
        if not self.loaded or roi.size == 0:
            return roi
            
        # Upscale using FSRCNN
        upscaled = self.sr.upsample(roi)
        
        return upscaled
