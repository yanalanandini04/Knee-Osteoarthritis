from src.rehab import EXERCISE_CONFIG, avatar_svg, recommendations


def test_recommendations_are_configured_for_all_grades():
    for grade in range(5):
        items = recommendations(grade)
        assert items
        assert all(item["name"] and "video_url" not in item for item in items)
        assert all(item["name"] in EXERCISE_CONFIG for item in items)
        assert all(grade in item["recommended_grades"] for item in items)
        assert all(item["avatar_movement"] for item in items)
        assert all(item["required_landmarks"] and item["joint_angles"] for item in items)
        assert all(item["acceptable_angle_range"] and item["repetition_logic"] for item in items)
        assert all({"missing_landmarks", "start", "complete", "range", "incorrect", "no_pose"} <= item["corrective_feedback"].keys() for item in items)


def test_recommendations_follow_severity_grade():
    grade_two = {item["name"] for item in recommendations(2)}
    grade_four = {item["name"] for item in recommendations(4)}

    assert "Supported sit-to-stand" in grade_two
    assert "Gentle seated range of motion" in grade_four
    assert grade_two != grade_four


def test_grade_recommendations_select_different_avatar_movements():
    grade_zero = recommendations(0)[0]
    grade_four = recommendations(4)[0]

    assert grade_zero["name"] == "Supported knee extension"
    assert grade_four["name"] == "Gentle seated range of motion"
    assert grade_zero["avatar_movement"] != grade_four["avatar_movement"]
    assert f'data-avatar-movement="{grade_zero["avatar_movement"]}"' in avatar_svg(grade_zero)
    assert f'data-avatar-movement="{grade_four["avatar_movement"]}"' in avatar_svg(grade_four)
    assert "<animateTransform" in avatar_svg(grade_zero)
    assert "<animateTransform" in avatar_svg(grade_four)
