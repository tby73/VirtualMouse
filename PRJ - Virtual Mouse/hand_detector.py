import mediapipe as mp
import cv2
import numpy as np

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5, complexity=1):
        # config variables
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence 
        self.tracking_confidence = tracking_confidence
        self.complexity = complexity

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.complexity, self.detection_confidence, self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

        self.TIP_IDS = [4, 8, 12, 16, 20]

    def FindHands(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        # process and display results
        if self.results.multi_hand_landmarks:
            for handlandmark in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, handlandmark, self.mp_hands.HAND_CONNECTIONS, 
                                            landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4), 
                                            connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        return image
    
    def GetLandmarkPositions(self, image, hand_index=0, draw=False):
        self.landmarklist = []

        # for bounding box construction
        x_list = []
        y_list = []
        bbox = []

        if self.results.multi_hand_landmarks:
            # only one hand has mouse control
            control_hand = self.results.multi_hand_landmarks[hand_index]

            for id, landmark in enumerate(control_hand.landmark):
                # get landmark coordinates
                height, width, channels = image.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)

                self.landmarklist.append([id, cx, cy])
                x_list.append(cx)
                y_list.append(cy)

            # construct bounding box
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            bbox = x_min, y_min, x_max, y_max

            if draw == True:
                cv2.rectangle(image, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0))

        return self.landmarklist, bbox
    
    def GetFingersUp(self):
        knuckle_ids = [2, 5, 9, 13, 17]

        # Initialize finger count
        finger_count = 0

        # Check each finger for being extended (upwards)
        for fingertip_id, knuckle_id in zip(self.TIP_IDS, knuckle_ids):
            fingertip_y = self.landmarklist[fingertip_id].y
            knuckle_y = self.landmarklist[knuckle_id].y

            # If the fingertip is higher (y-coordinate) than the knuckle, consider it extended
            if fingertip_y < knuckle_y:
                finger_count += 1

        return finger_count

