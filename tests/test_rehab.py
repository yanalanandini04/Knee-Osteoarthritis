from src.rehab import recommendations


def test_recommendations_are_configured_for_all_grades():
    for grade in range(5):
        items = recommendations(grade)
        assert items
        assert all(item["name"] and item["video_url"] for item in items)


def test_recommendations_follow_severity_grade():
    grade_two = {item["name"] for item in recommendations(2)}
    grade_four = {item["name"] for item in recommendations(4)}

    assert "Supported sit-to-stand" in grade_two
    assert "Gentle seated range of motion" in grade_four
    assert grade_two != grade_four
