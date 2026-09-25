from types import SimpleNamespace

from src.monitor import MonitorState, angle, assess_knee_flexion


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
