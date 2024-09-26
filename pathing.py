import logging
from abc import ABC, abstractmethod

import numpy as np
from pygame import Vector2
from scipy.interpolate import splev, splprep

from ...constants import framerate

class Path(ABC):
    def __init__(self, waypoints: list[Vector2], n_rounds: int):
        assert n_rounds > 0
        self.n_rounds = n_rounds

        coords = self.fit(waypoints * n_rounds + [waypoints[0]])
        assert isinstance(coords, np.ndarray)
        assert coords.shape[1] == 2
        assert coords.shape[0] > 10

        self.coords = coords
        # self.start_coords, self.cycle_coords = self._generate_rounds(coords)

    @abstractmethod
    def fit(self, waypoints: list[Vector2]):
        pass

    def get_coords(self, cp: int, limit: int) -> np.ndarray:
        if limit == 0:
            return self.get_coords[cp:]
        else:
            return self.coords[range(cp, min(cp + limit, self.coords.shape[0])), :]

    def get_node(self, cp: int):
        return Vector2(*self.coords[cp, :])

    def get_nodes(self, **kwargs) -> list[Vector2]:
        coords = self.get_coords(**kwargs)
        return [Vector2(float(x), float(y)) for x, y in zip(coords[:, 0], coords[:, 1])]

    def get_last_checkpoint(
        self, pos: Vector2, velocity: Vector2, cp: int = 0, limit: int = 0
    ) -> int:
        next_pos = pos + velocity / framerate
        candidate_coords = self.get_coords(cp=cp, limit=limit)
        distances = np.linalg.norm(candidate_coords - np.array([next_pos[0], next_pos[1]]), axis=1)
        nearest_index = np.argmin(distances)

        if nearest_index == 0:
            return cp

        logging.info(f'Approaching new checkpoint {cp + nearest_index}')
        # check if we've crossed into a next segment
        nearest_node = Vector2(*candidate_coords[nearest_index])
        next_node = Vector2(*candidate_coords[nearest_index + 1])
        start_diff = nearest_node - pos
        segment_diff = nearest_node - next_node
        in_segment = 0 < start_diff.dot(segment_diff) / segment_diff.dot(segment_diff) < 1

        if in_segment:
            logging.info('Entered next segment!')
            return cp + nearest_index
        else:
            logging.info('Entered new segment!')
            return cp + nearest_index - 1

    def _generate_rounds(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # find transition checkpoint
        distances = np.linalg.norm(coords[:coords.shape[0] / 2, :] - coords[-1, :], axis=1)
        trans_index = np.argmin(distances)

        coords1 = coords
        coords2 = coords[trans_index:, ]

        return coords1, coords2


class BsPath(Path):
    def __init__(self, res: int, **kwargs):
        assert res >= 10
        self.res = res
        super().__init__(**kwargs)

    def fit(self, waypoints: list[Vector2]):
        x = [p[0] for p in waypoints]
        y = [p[1] for p in waypoints]

        tck, u = splprep([x, y], s=0, k=2)

        u_new = np.linspace(0, 1, self.res)
        x_new, y_new = splev(u_new, tck)

        coords = np.column_stack([x_new, y_new])
        return coords
