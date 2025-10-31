max_loop = 5
from dataclasses import dataclass

@dataclass
class Region:
    id: int
    coords: tuple[float, float]
    area: float

@dataclass
class UAV:
    id: int
    max_velocity: float
    scan_width: float

BASE_COORDS = (0.0, 0.0)