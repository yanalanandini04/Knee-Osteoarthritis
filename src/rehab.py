import json
from pathlib import Path


def load_raw_config(path: str = "config/exercises.json") -> dict:
    file_path = Path(path)
    if not file_path.exists():
        file_path = Path(__file__).resolve().parent.parent / "config" / "exercises.json"
    with file_path.open(encoding="utf-8") as file:
        return json.load(file)


_raw = load_raw_config()

if "exercises" in _raw:
    EXERCISE_CONFIG = dict(_raw["exercises"])
    GRADE_MAP = dict(_raw.get("grade_mapping", {}))
else:
    EXERCISE_CONFIG = dict(_raw)
    GRADE_MAP = {
        "0": ["Supported knee extension", "Seated heel slide"],
        "1": ["Supported knee extension", "Standing hamstring curl"],
        "2": ["Supported sit-to-stand", "Standing hamstring curl"],
        "3": ["Supported weight shift", "Gentle seated range of motion"],
        "4": ["Gentle seated range of motion"],
    }


def load_exercises(path: str = "config/exercises.json") -> dict:
    raw = load_raw_config(path)
    return raw.get("exercises", EXERCISE_CONFIG)


def avatar_svg(exercise_or_movement: dict | str = "seated_knee_extension") -> str:
    """Return avatar SVG definition matching the exercise avatar movement."""
    if isinstance(exercise_or_movement, dict):
        movement = exercise_or_movement.get("avatar_movement", "seated_knee_extension")
    else:
        movement = str(exercise_or_movement or "seated_knee_extension")

    return f"""
    <div style='margin-top: 1rem; background: linear-gradient(180deg, #0f172a, #111827); border-radius: 16px; padding: 1rem; text-align: center;'>
      <svg viewBox='0 0 260 260' width='100%' height='220' xmlns='http://www.w3.org/2000/svg' data-avatar-movement="{movement}">
        <g stroke='white' stroke-width='6' stroke-linecap='round' fill='none'>
          <!-- Head -->
          <circle cx='130' cy='38' r='20' stroke='#38bdf8' fill='#0284c7'/>
          <!-- Spine/Torso -->
          <line x1='130' y1='58' x2='130' y2='120' stroke='#93c5fd'/>
          <!-- Arms -->
          <line x1='130' y1='75' x2='95' y2='105' stroke='#c4b5fd'/>
          <line x1='130' y1='75' x2='165' y2='105' stroke='#c4b5fd'/>
          <!-- Pelvis / Left Hip -->
          <line x1='130' y1='120' x2='105' y2='165' stroke='#f472b6'/>
          <line x1='105' y1='165' x2='105' y2='215' stroke='#f472b6'/>
          <!-- Active Right Leg with animated lower limb -->
          <line x1='130' y1='120' x2='155' y2='165' stroke='#34d399'/>
          <g transform='translate(155, 165)'>
            <line x1='0' y1='0' x2='0' y2='50' stroke='#10b981'>
              <animateTransform attributeName='transform' type='rotate'
                values='0; -55; 0' dur='2.5s' repeatCount='indefinite'
                keyTimes='0; 0.5; 1' calcMode='spline' keySplines='0.4 0 0.2 1; 0.4 0 0.2 1' />
            </line>
          </g>
        </g>
      </svg>
    </div>
    """


def recommendations(grade: int, exercise_map: dict | None = None) -> list[dict]:
    raw = load_raw_config()
    grade_map = raw.get("grade_mapping", GRADE_MAP)
    names = grade_map.get(str(grade), [])
    cfg = raw.get("exercises", EXERCISE_CONFIG) if exercise_map is None else exercise_map
    results = []
    for n in names:
        if n in cfg:
            results.append(cfg[n])
        elif isinstance(cfg, dict):
            for sub in cfg.values():
                if isinstance(sub, dict) and sub.get("name") == n:
                    results.append(sub)
    return results