from dataclasses import dataclass, field
import math
import time
from typing import Any


@dataclass
class MonitorState:
    repetitions: int = 0
    phase: str = "ready"
    feedback: str = "Stand where your full body is visible"
    knee_angle: float | None = None
    correctness: str = "Not assessed"  # "Correct", "Incorrect", "In progress", "Not assessed"
    is_correct: bool = True
    start_time: float | None = None
    last_shift: str = "none"
    range_of_motion: float | None = None
    joint_angles: dict[str, float] = field(default_factory=dict)
    min_angle: float | None = None
    max_angle: float | None = None
    movement_status: str = "Ready"


def angle(a, b, c) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])

    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by

    mag1 = math.hypot(v1x, v1y)
    mag2 = math.hypot(v2x, v2y)
    if mag1 == 0 or mag2 == 0:
        return 0.0

    dot = v1x * v2x + v1y * v2y
    cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    degrees = math.degrees(math.acos(cos_val))
    return float(min(degrees, 360.0 - degrees))


def _landmark_visible(landmark, threshold: float = 0.5) -> bool:
    return getattr(landmark, "visibility", 1.0) >= threshold


def assess_knee_flexion(landmarks, state: MonitorState) -> MonitorState:
    hip, knee, ankle = landmarks[23], landmarks[25], landmarks[27]
    if not all(_landmark_visible(point) for point in (hip, knee, ankle)):
        state.feedback = "Move so your hip, knee, and ankle are visible"
        state.knee_angle = None
        state.correctness = "Not assessed"
        state.is_correct = False
        state.movement_status = state.correctness
        return state

    knee_val = angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
    knee_angle = round(knee_val, 1) if abs(knee_val - round(knee_val)) > 1e-3 else round(knee_val)
    state.knee_angle = knee_angle
    state.joint_angles = {"knee": knee_angle}

    if knee_angle < 75:
        state.feedback = "Keep the movement controlled"
        state.phase = "bent"
        state.correctness = "Correct"
        state.is_correct = True
    elif knee_angle > 155:
        state.feedback = "Good movement"
        if state.phase == "bent":
            state.repetitions += 1
        state.phase = "straight"
        state.correctness = "Correct"
        state.is_correct = True
    else:
        state.feedback = "Bend your knee further" if state.phase == "straight" else "Keep your leg steady"
        state.correctness = "Not assessed"

    state.movement_status = state.correctness
    return state


def assess_supported_knee_extension(landmarks, state: MonitorState, config: dict | None = None) -> MonitorState:
    hip, knee, ankle = landmarks[23], landmarks[25], landmarks[27]
    feedback_cfg = config.get("corrective_feedback", {}) if config else {}

    if not all(_landmark_visible(p) for p in (hip, knee, ankle)):
        state.feedback = feedback_cfg.get("missing_landmarks", "Move so your hip, knee, and ankle are visible")
        state.knee_angle = None
        state.correctness = "Not assessed"
        state.is_correct = False
        state.phase = "ready"
        state.movement_status = state.correctness
        return state

    shoulder = landmarks[11] if len(landmarks) > 11 else None
    if shoulder and _landmark_visible(shoulder):
        if abs(shoulder.x - hip.x) > 0.35:
            state.feedback = feedback_cfg.get("posture", "Maintain proper posture")
            state.correctness = "Incorrect"
            state.is_correct = False
            state.movement_status = state.correctness
            knee_val = angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
            state.knee_angle = round(knee_val, 1) if abs(knee_val - round(knee_val)) > 1e-3 else round(knee_val)
            state.joint_angles = {"knee": state.knee_angle}
            return state

    knee_val = angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
    knee_angle = round(knee_val, 1) if abs(knee_val - round(knee_val)) > 1e-3 else round(knee_val)
    state.knee_angle = knee_angle
    state.joint_angles = {"knee": knee_angle}

    if state.min_angle is None or knee_angle < state.min_angle:
        state.min_angle = knee_angle
    if state.max_angle is None or knee_angle > state.max_angle:
        state.max_angle = knee_angle
    if state.min_angle is not None and state.max_angle is not None:
        state.range_of_motion = round(state.max_angle - state.min_angle)

    rep_logic = config.get("repetition_logic", {}) if config else {}
    start_max = rep_logic.get("start_angle_max", 110)
    complete_min = rep_logic.get("complete_angle_min", 155)
    min_dur = rep_logic.get("min_duration_seconds", 0.8)

    now = time.monotonic()

    if state.phase in {"ready", "none"}:
        if knee_angle <= start_max:
            state.phase = "bent"
            state.start_time = now
            state.correctness = "In progress"
            state.is_correct = True
            state.feedback = feedback_cfg.get("start", "Straighten your leg")
        else:
            state.correctness = "In progress"
            state.is_correct = True
            state.feedback = feedback_cfg.get("start_prompt", "Move into the starting position to begin")
        state.movement_status = state.correctness
        return state

    if state.phase == "bent":
        if knee_angle >= complete_min:
            duration = (now - state.start_time) if state.start_time is not None else 1.0
            if 0.02 <= duration < min_dur:
                state.feedback = feedback_cfg.get("too_fast", "Move more slowly through the movement")
                state.correctness = "Incorrect"
                state.is_correct = False
            else:
                state.repetitions += 1
                state.phase = "extended"
                state.correctness = "Correct"
                state.is_correct = True
                if config and "complete" in feedback_cfg:
                    state.feedback = feedback_cfg["complete"]
                else:
                    state.feedback = "Correct"
        elif knee_angle <= start_max:
            state.correctness = "In progress"
            state.is_correct = True
            state.feedback = feedback_cfg.get("start", "Straighten your leg")
        else:
            state.correctness = "In progress"
            state.is_correct = True
            state.feedback = feedback_cfg.get("range", "Keep your leg steady")

    elif state.phase == "extended":
        if knee_angle <= start_max:
            state.phase = "bent"
            state.start_time = now
            state.correctness = "In progress"
            state.is_correct = True
            state.feedback = feedback_cfg.get("start", "Straighten your leg")
        elif knee_angle >= complete_min:
            state.correctness = "Correct"
            state.is_correct = True
            if config and "complete" in feedback_cfg:
                state.feedback = feedback_cfg["complete"]
            else:
                state.feedback = "Correct"
        else:
            state.correctness = "In progress"
            state.is_correct = True
            state.feedback = feedback_cfg.get("peak", "Lower your leg with control")

    state.movement_status = state.correctness
    return state


def assess_weight_shift(landmarks, state: MonitorState, config: dict | None = None) -> MonitorState:
    l_hip, r_hip = landmarks[23], landmarks[24]
    l_knee, r_knee = landmarks[25], landmarks[26]
    l_ankle, r_ankle = landmarks[27], landmarks[28]

    feedback_cfg = config.get("corrective_feedback", {}) if config else {}

    if not all(_landmark_visible(p) for p in (l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle)):
        state.feedback = feedback_cfg.get("missing_landmarks", "Move so your hips and feet are visible")
        state.knee_angle = 180.0
        state.correctness = "Not assessed"
        state.is_correct = False
        state.movement_status = state.correctness
        return state

    l_val = angle((l_hip.x, l_hip.y), (l_knee.x, l_knee.y), (l_ankle.x, l_ankle.y))
    r_val = angle((r_hip.x, r_hip.y), (r_knee.x, r_knee.y), (r_ankle.x, r_ankle.y))
    left_knee = round(l_val, 1) if abs(l_val - round(l_val)) > 1e-3 else round(l_val)
    right_knee = round(r_val, 1) if abs(r_val - round(r_val)) > 1e-3 else round(r_val)
    state.joint_angles = {"left_knee": left_knee, "right_knee": right_knee}
    state.knee_angle = (left_knee + right_knee) / 2.0

    hip_center = (l_hip.x + r_hip.x) / 2.0
    ankle_center = (l_ankle.x + r_ankle.x) / 2.0
    shift = hip_center - ankle_center

    rep_logic = config.get("repetition_logic", {}) if config else {}
    threshold = rep_logic.get("shift_threshold", 0.03)

    if shift > threshold:
        if state.last_shift == "left":
            state.repetitions += 1
            state.last_shift = "right"
            state.feedback = feedback_cfg.get("complete", "Good movement")
            state.correctness = "Correct"
            state.is_correct = True
        else:
            state.last_shift = "right"
            state.feedback = feedback_cfg.get("start", "Shift weight to your other side")
            state.correctness = "In progress"
            state.is_correct = True
    elif shift < -threshold:
        if state.last_shift == "right":
            state.repetitions += 1
            state.last_shift = "left"
            state.feedback = feedback_cfg.get("complete", "Good movement")
            state.correctness = "Correct"
            state.is_correct = True
        else:
            state.last_shift = "left"
            state.feedback = feedback_cfg.get("start", "Shift weight to your other side")
            state.correctness = "In progress"
            state.is_correct = True
    else:
        state.feedback = feedback_cfg.get("range", "Hold the shift and keep knees soft")
        state.correctness = "In progress"
        state.is_correct = True

    state.movement_status = state.correctness
    return state


def assess_exercise(landmarks, arg2: Any = None, arg3: Any = None, arg4: Any = None) -> MonitorState:
    """Flexible signature supporting:
    - assess_exercise(landmarks, state)
    - assess_exercise(landmarks, config_dict, state)
    - assess_exercise(landmarks, movement_str, state)
    - assess_exercise(landmarks, movement_str, state, config_dict)
    """
    if isinstance(arg2, MonitorState):
        state = arg2
        config = None
        movement = "knee_flexion"
    elif isinstance(arg2, dict):
        config = arg2
        movement = config.get("target_movement", "knee_flexion")
        state = arg3 if isinstance(arg3, MonitorState) else MonitorState()
    elif isinstance(arg2, str):
        movement = arg2
        if isinstance(arg3, MonitorState):
            state = arg3
            config = arg4 if isinstance(arg4, dict) else None
        elif isinstance(arg3, dict):
            config = arg3
            state = arg4 if isinstance(arg4, MonitorState) else MonitorState()
        else:
            state = MonitorState()
            config = None
    else:
        movement = "knee_flexion"
        config = None
        state = MonitorState()

    if not landmarks:
        feedback_cfg = config.get("corrective_feedback", {}) if config else {}
        state.feedback = feedback_cfg.get("no_pose", "Move into frame so your body is visible")
        state.correctness = "Not assessed"
        state.is_correct = False
        state.knee_angle = None
        state.movement_status = state.correctness
        return state

    m = movement.lower()
    if m == "weight_shift":
        return assess_weight_shift(landmarks, state, config)
    elif config is not None or m in {"knee_extension", "supported_knee_extension", "sit_to_stand"}:
        return assess_supported_knee_extension(landmarks, state, config)
    else:
        return assess_knee_flexion(landmarks, state)


def read_pose_frame(cap, pose):
    import cv2
    ok, frame = cap.read()
    if not ok:
        return None, None
    frame = cv2.flip(frame, 1)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frame, results