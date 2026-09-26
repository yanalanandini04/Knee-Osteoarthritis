from types import SimpleNamespace

from src.monitor import MonitorState, angle, assess_exercise, assess_knee_flexion, assess_supported_knee_extension
from src.rehab import EXERCISE_CONFIG, avatar_svg, recommendations


def side_view_landmarks(knee_angle, shoulder=(0, -1), visibility=1.0):
    import math

    landmarks = [SimpleNamespace(x=0, y=0, visibility=visibility) for _ in range(33)]
    radians = math.radians(180 - knee_angle)
    landmarks[11] = SimpleNamespace(x=shoulder[0], y=shoulder[1], visibility=visibility)
    landmarks[23] = SimpleNamespace(x=0, y=0, visibility=visibility)
    landmarks[25] = SimpleNamespace(x=1, y=0, visibility=visibility)
    landmarks[27] = SimpleNamespace(x=1 + math.cos(radians), y=math.sin(radians), visibility=visibility)
    return landmarks


def test_angle_is_expected():
    assert round(angle((0, 0), (0, 1), (1, 1))) == 90


def test_bent_then_straight_counts_one_rep():
    state = MonitorState()
    bent = [SimpleNamespace(x=0, y=0) for _ in range(33)]
    bent[23], bent[25], bent[27] = SimpleNamespace(x=0, y=0), SimpleNamespace(x=0, y=1), SimpleNamespace(x=0.5, y=0.5)
    assess_knee_flexion(bent, state)
    straight = [SimpleNamespace(x=0, y=0) for _ in range(33)]
    straight[23], straight[25], straight[27] = SimpleNamespace(x=0, y=0), SimpleNamespace(x=0, y=1), SimpleNamespace(x=0, y=2)
    assess_knee_flexion(straight, state)
    assert state.repetitions == 1


def test_low_visibility_does_not_count_repetition():
    state = MonitorState()
    landmarks = [SimpleNamespace(x=0, y=0, visibility=1.0) for _ in range(33)]
    landmarks[23] = SimpleNamespace(x=0, y=0, visibility=0.2)

    assess_knee_flexion(landmarks, state)

    assert state.repetitions == 0
    assert state.knee_angle is None
    assert "visible" in state.feedback


def test_supported_knee_extension_checks_posture_rom_and_repetitions():
    state = MonitorState()

    assess_supported_knee_extension(side_view_landmarks(100), state)
    assert state.feedback == "Straighten your leg"
    assert state.phase == "bent"
    assert state.correctness == "In progress"

    assess_supported_knee_extension(side_view_landmarks(170), state)
    assert state.feedback == "Correct"
    assert state.correctness == "Correct"
    assert state.repetitions == 1
    assert state.range_of_motion == 70

    assess_supported_knee_extension(side_view_landmarks(170), state)
    assert state.feedback == "Correct"

    assess_supported_knee_extension(side_view_landmarks(100, shoulder=(1, -1)), state)
    assert state.feedback == "Maintain proper posture"
    assert state.correctness == "Incorrect"
    assert state.repetitions == 1

    assess_supported_knee_extension(side_view_landmarks(100), state)
    assess_supported_knee_extension(side_view_landmarks(170), state)
    assert state.repetitions == 2


def test_supported_knee_extension_does_not_count_without_full_cycle():
    state = MonitorState()

    assess_supported_knee_extension(side_view_landmarks(170), state)
    assess_supported_knee_extension(side_view_landmarks(100, visibility=0.2), state)
    assert "visible" in state.feedback
    assert state.correctness == "Not assessed"
    assess_supported_knee_extension(side_view_landmarks(170), state)

    assert state.repetitions == 0


def test_every_configured_knee_exercise_counts_a_complete_repetition():
    for config in EXERCISE_CONFIG.values():
        if config["target_movement"] == "weight_shift":
            continue
        state = MonitorState()

        assess_exercise(side_view_landmarks(100), config, state)
        assess_exercise(side_view_landmarks(170), config, state)

        assert state.repetitions == 1, config["name"]
        assert state.correctness == "Correct", config["name"]
        assert state.joint_angles["knee"] == 170


def test_grade_recommendation_avatar_to_live_feedback_cycle():
    exercise = recommendations(0)[0]
    state = MonitorState()

    assert exercise["name"] == "Supported knee extension"
    assert "video_url" not in exercise
    assert exercise["avatar_movement"] == "seated_knee_extension"
    assert exercise["joint_angles"]["knee"] == ["hip", "knee", "ankle"]
    assess_exercise(side_view_landmarks(100), exercise, state)
    assess_exercise(side_view_landmarks(170), exercise, state)

    assert state.repetitions == 1
    assert state.knee_angle == 170
    assert state.correctness == "Correct"
    assert state.feedback == "Good movement"

    assess_exercise(side_view_landmarks(100, shoulder=(1, -1)), exercise, state)
    assert state.correctness == "Incorrect"
    assert state.feedback == exercise["corrective_feedback"]["posture"]


def test_two_grade_rehab_flows_use_grade_specific_avatars_and_motion_rules():
    for grade, expected_name, expected_avatar in (
        (0, "Supported knee extension", "seated_knee_extension"),
        (4, "Gentle seated range of motion", "gentle_seated_range_of_motion"),
    ):
        exercise = recommendations(grade)[0]
        state = MonitorState()

        assert exercise["name"] == expected_name
        assert exercise["avatar_movement"] == expected_avatar
        assert f'data-avatar-movement="{expected_avatar}"' in avatar_svg(exercise)
        assess_exercise(side_view_landmarks(100), exercise, state)
        assess_exercise(side_view_landmarks(170), exercise, state)

        assert state.repetitions == 1
        assert state.correctness == "Correct"
        assert state.feedback == "Good movement"


def test_assessor_handles_missing_corrective_feedback_configuration():
    config = {
        "target_movement": "knee_flexion",
        "required_landmarks": ["hip", "knee", "ankle"],
        "joint_angles": {"knee": ["hip", "knee", "ankle"]},
        "repetition_logic": {"start_angle_max": 120, "complete_angle_min": 155},
    }
    state = assess_exercise(side_view_landmarks(170), config, MonitorState())

    assert state.correctness == "In progress"
    assert state.feedback == "Move into the starting position to begin"


def test_fast_knee_movement_returns_corrective_feedback(monkeypatch):
    timestamps = iter((1.0, 1.2))
    monkeypatch.setattr("src.monitor.time.monotonic", lambda: next(timestamps))
    exercise = recommendations(0)[0]
    state = MonitorState()

    assess_exercise(side_view_landmarks(100), exercise, state)
    assess_exercise(side_view_landmarks(170), exercise, state)

    assert state.correctness == "Incorrect"
    assert state.feedback == "Move more slowly through the movement"
    assert state.repetitions == 0


def test_weight_shift_counts_alternating_shifts_and_checks_knee_angles():
    config = EXERCISE_CONFIG["Supported weight shift"]
    landmarks = [SimpleNamespace(x=0, y=0, visibility=1.0) for _ in range(33)]
    for side, hip_index, knee_index, ankle_index, x in (
        ("left", 23, 25, 27, 0.4),
        ("right", 24, 26, 28, 0.6),
    ):
        landmarks[hip_index] = SimpleNamespace(x=x, y=0, visibility=1.0)
        landmarks[knee_index] = SimpleNamespace(x=x, y=0.5, visibility=1.0)
        landmarks[ankle_index] = SimpleNamespace(x=x, y=1, visibility=1.0)

    state = MonitorState()
    assess_exercise(landmarks, config, state)
    for index, hip_x in ((23, 0.3), (24, 0.5)):
        landmarks[index].x = hip_x
    assess_exercise(landmarks, config, state)
    for index, hip_x in ((23, 0.5), (24, 0.7)):
        landmarks[index].x = hip_x
    assess_exercise(landmarks, config, state)

    assert state.repetitions == 1
    assert state.correctness == "Correct"
    assert set(state.joint_angles) == {"left_knee", "right_knee"}
