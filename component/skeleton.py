import mediapipe as mp
import numpy as np
import json


def skeleton(image):
    mp_pose = mp.solutions.pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    solution = mp_pose.process(image)

    # default value
    result = np.zeros((33, 2))

    # detect and draw skeleton
    if solution.pose_landmarks:
        mp_drawing.draw_landmarks(image, solution.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        result = np.array([
            (landmark.y, landmark.x)
            for landmark in solution.pose_landmarks.landmark
        ])

    # serialization
    skeleton_info = json.dumps(np.array([result[15:22:2], result[16:23:2]]).tolist())

    return skeleton_info, image