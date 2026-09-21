import json
from pathlib import Path


def load_exercises(path: str = "config/exercises.json") -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def recommendations(grade: int, exercise_map: dict | None = None) -> list[dict]:
    exercise_map = load_exercises() if exercise_map is None else exercise_map
    return exercise_map.get(str(grade), [])
