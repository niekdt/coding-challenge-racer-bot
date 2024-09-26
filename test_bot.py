import math

import pytest
from pygame import Vector2

from .bot import MinVerstappen, get_checkpoints
from .physics import compute_local_turning_radius

def test_init():
    MinVerstappen(track=None)


@pytest.mark.parametrize('radius', [1, 2, 50, 100, 200, 500, 1000])
@pytest.mark.parametrize('dtheta', [.001, .01, .1, 1, 5, 10, 15])
def test_compute_turn_radius(radius: float, dtheta: float):
    prev_pos = Vector2.from_polar((radius, 0))
    pos = Vector2.from_polar((radius, dtheta))
    next_pos = Vector2.from_polar((radius, dtheta * 2))
    r = compute_local_turning_radius(prev_pos, pos, next_pos)
    assert r > 0
    assert math.isclose(r, radius, rel_tol=.005)

@pytest.mark.parametrize('angle', range(0, 361, 15))
@pytest.mark.parametrize('distance', [.01, .1, 1, 10, 100])
def test_compute_turn_radius_straight(angle: float, distance: float):
    pos = Vector2()
    prev_pos = Vector2.from_polar((distance, -angle))
    next_pos = Vector2.from_polar((distance, 180 - angle))
    r = compute_local_turning_radius(prev_pos, pos, next_pos)
    assert r > 1e5


def test_compute_turn_radius_flip():
    x = [424.63, 417.01, 409.39]
    y = [919.9 , 919.93, 919.89]

    prev_pos = Vector2(x[0], y[0])
    pos = Vector2(x[1], y[1])
    next_pos = Vector2(x[2], y[2])

    r = compute_local_turning_radius(prev_pos, pos, next_pos)
    assert r > 100


def test_get_checkpoints():
    assert get_checkpoints(last=0, limit=1) == [0]
    assert get_checkpoints(last=1, limit=1) == [1]
    assert get_checkpoints(last=1, limit=2) == [1, 2]
    assert get_checkpoints(last=1, limit=2, res=2) == [1, 2]
    assert get_checkpoints(last=1, limit=2, res=2, n_trans=1) == [1, 2]
    assert get_checkpoints(last=2, limit=2, res=2, n_trans=1) == [2, 1]
    assert get_checkpoints(last=2, limit=5, res=2, n_trans=1) == [2, 1, 2, 1, 2]
    assert get_checkpoints(last=4, limit=3, res=5, n_trans=1) == [4, 5, 1]
    assert get_checkpoints(last=4, limit=5, res=5, n_trans=1) == [4, 5, 6, 2, 3]
    assert get_checkpoints(last=4, limit=5, res=5, n_trans=2) == [4, 5, 6, 7, 3]
    assert get_checkpoints(last=4, limit=5, res=5, n_trans=3) == [4, 5, 6, 7, 8]
    assert get_checkpoints(last=4, limit=7, res=5, n_trans=3) == [4, 5, 6, 7, 8, 4, 5]
