import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pg
import screeninfo

from hand_detector import HandDetector
from face_detector import FaceDetector
from genderagedetector import GenderAgeDetector
from visualization_utils import BoxDisplay

# camera indexes
DEFAULT_CAMERA = 0
SECONDARY_CAMERA = 1

# max n of hands allowed for use
HANDS_LIMIT = 1

# thumb and index tip
THUMB_TIP_ID = 4
INDEX_TIP_ID = 8

# face detector model path
FACE_DETECTOR_MODEL_PATH = "haarcascade_frontalface_default.xml"

# get screen size
screen_info = screeninfo.get_monitors()[0]  # Get the primary monitor information
screen_width = screen_info.width
screen_height = screen_info.height

def main():
    # init video capture and hand detector
    video_cap = cv2.VideoCapture(DEFAULT_CAMERA)
    hand_detector = HandDetector(max_hands=HANDS_LIMIT)

    while True:
        _, frame = video_cap.read()
        frame = hand_detector.FindHands(frame)
        landmarkpositions, bbox = hand_detector.GetLandmarkPositions(frame, draw=True)
        
        fingers_up = hand_detector.GetFingersUp(frame)
        click_length, [x1, y1, x2, y2, cx, cy] = hand_detector.ClickDistance(frame, THUMB_TIP_ID, INDEX_TIP_ID, 15, 3, draw=True)

        # init face detector
        face_detector = FaceDetector(frame, FACE_DETECTOR_MODEL_PATH)
        faces = face_detector.DetectFaces()

        # init gender and age recognizer
        ga_detector = GenderAgeDetector(frame)

        # display all
        for x, y, w, h in faces:
            cropped = ga_detector.Crop(x, y, w, h)
            pred_gender, pred_age = ga_detector.GetResults(cropped)
            BoxDisplay.BoundingBoxwithMultilineInfo(frame, 2, [f"Gender: {pred_gender}", f"Age: {pred_age}"], 0.62, x, y, w, h, color=(0, 255, 0), thickness=2)
        
        # detect if hand is in canvas and perform click if finger index 4 and 8 are close
        if 0 < cx < screen_width and 0 < cy < screen_height:
            corrected_cx = screen_width - cx  # Correcting the x-coordinate

            pg.moveTo(corrected_cx, cy)

            if click_length < 20:
                pg.click()

        cv2.imshow("Virtual Mouse v1.0", frame)

        # quit
        if cv2.waitKey(20) & 0xff == ord("q"):
            break

    video_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

