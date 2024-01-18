import cv2
import numpy as np

class FaceDetector:
    def __init__(self, image, classifier_path) -> None:
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.classifier_path = classifier_path
        self.detector = cv2.CascadeClassifier(self.classifier_path)

    def DetectFaces(self):
        return self.detector.detectMultiScale(self.image, 1.1, minNeighbors=5, minSize=(40, 40))
    

