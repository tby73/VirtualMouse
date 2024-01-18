import cv2
import numpy as np

from deepface import DeepFace

class GenderAgeDetector:
    def __init__(self, image) -> None:
        self.image = image

    def Crop(self, x, y, w, h):
        return self.image[x:w, y:h]
    
    def GetResults(self, cropped):
        results = DeepFace.analyze(self.image, actions=("gender", "age"), enforce_detection=False)
        return results[0]["dominant_gender"], results[0]["age"]
    
