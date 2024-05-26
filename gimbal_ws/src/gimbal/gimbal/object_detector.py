from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_objects(self, image):
        return self.model(image, verbose=False, classes=67)[0]