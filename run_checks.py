from src.monitor import MonitorState, angle
from src.rehab import recommendations

assert round(angle((0, 0), (0, 1), (1, 1))) == 90
assert all(recommendations(grade) for grade in range(5))
print("Core checks passed")
