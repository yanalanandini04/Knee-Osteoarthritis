from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MonitorState:
    repetitions: int = 0
    phase: str = "ready"
    feedback: str = "Stand where your full body is visible"


def angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    return float(abs(np.degrees(radians)) % 360)


def assess_knee_flexion(landmarks, state: MonitorState) -> MonitorState:
    hip, knee, ankle = landmarks[23], landmarks[25], landmarks[27]
    knee_angle = angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
    if knee_angle < 75:
        state.feedback = "Keep the movement controlled"
        state.phase = "bent"
    elif knee_angle > 155:
        state.feedback = "Good movement"
        if state.phase == "bent":
            state.repetitions += 1
        state.phase = "straight"
    else:
        state.feedback = "Bend your knee further" if state.phase == "straight" else "Keep your leg steady"
    return state


def assess_exercise(landmarks, movement: str, state: MonitorState) -> MonitorState:
    if movement in {"knee_flexion", "knee_extension", "sit_to_stand"}:
        return assess_knee_flexion(landmarks, state)
    state.feedback = "Good movement" if landmarks else "Keep your body visible"
    return state


def read_pose_frame(cap, pose):
    import cv2

    ok, frame = cap.read()
    if not ok:
        return None, None
    frame = cv2.flip(frame, 1)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frame, results
