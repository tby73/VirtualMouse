import cv2
import mediapipe as mp
import numpy as np

from hand_detector import HandDetector

# camera indexes
DEFAULT_CAMERA = 0
SECONDARY_CAMERA = 1

# max n of hands allowed for use
HANDS_LIMIT = 1

# hand landmark positions
INDEX_FINGER_TOP = 8

def main():
    video_cap = cv2.VideoCapture(DEFAULT_CAMERA)
    hand_detector = HandDetector(max_hands=HANDS_LIMIT)

    while True:
        _, frame = video_cap.read()
        frame = hand_detector.FindHands(frame)
        landmarkpositions, bbox = hand_detector.GetLandmarkPositions(frame, draw=True)

        if len(landmarkpositions) != 0:
            x1, y1 = landmarkpositions[8][1:]
            x2, y2 = landmarkpositions[12][1:]
            # print(x1, y1, x2, y2)

        cv2.imshow("Virtual Mouse v1.0", frame)

        # quit
        if cv2.waitKey(20) & 0xff == ord("q"):
            break

    video_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

