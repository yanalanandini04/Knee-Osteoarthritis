from src.rehab import recommendations


def test_recommendations_are_configured_for_all_grades():
    for grade in range(5):
        items = recommendations(grade)
        assert items
        assert all(item["name"] and item["video_url"] for item in items)
