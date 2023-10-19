import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pg
import screeninfo

from hand_detector import HandDetector

# camera indexes
DEFAULT_CAMERA = 0
SECONDARY_CAMERA = 1

# max n of hands allowed for use
HANDS_LIMIT = 1

# get screen size
screen_info = screeninfo.get_monitors()[0]  # Get the primary monitor information
screen_width = screen_info.width
screen_height = screen_info.height

def main():
    video_cap = cv2.VideoCapture(DEFAULT_CAMERA)
    hand_detector = HandDetector(max_hands=HANDS_LIMIT)

    while True:
        _, frame = video_cap.read()
        frame = hand_detector.FindHands(frame)
        landmarkpositions, bbox = hand_detector.GetLandmarkPositions(frame, draw=True)
        
        fingers_up = hand_detector.GetFingersUp(frame)
        click_length, [x1, y1, x2, y2, cx, cy] = hand_detector.ClickDistance(frame, 4, 8, 15, 3, draw=True)
        
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
