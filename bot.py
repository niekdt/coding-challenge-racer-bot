import logging
import math
import os

import numpy as np

from typing import Tuple
from pygame import Vector2
from itertools import cycle, islice, pairwise

from track import Track
from ...bot import Bot
from ...linear_math import Transform

MAX_ACCELERATION = 5.0 / 3.0
MAX_DECELERATION = -MAX_ACCELERATION
MIN_SPEED = 150

if os.getenv('LOG'):
    logging.basicConfig(level='INFO')


class MinVerstappen(Bot):
    def __init__(self, track: Track):
        super().__init__(track)
        self.prev_pos = Vector2()
        self.prev_speed = 0

    @property
    def name(self):
        return "Min Verstappen"

    @property
    def contributor(self):
        return "Niek"

    def compute_commands(self, next_waypoint: int, position: Transform, velocity: Vector2) -> Tuple:
        def get_lines(offset=0, limit=20):
            start = (next_waypoint + offset) % len(self.track.lines)
            return list(islice(cycle(self.track.lines), start, start + limit))

        pos = position.p
        speed = velocity.length()
        acc = speed - self.prev_speed  # t = 1
        next_target = self.track.lines[next_waypoint]
        targets = get_lines()
        target_distance = next_target.distance_to(pos)
        target_angle = turn_angle(waypoint_angle(next_target, pos, targets[1]))

        logging.info(f'NEXT WAYPOINT: {next_waypoint} (distance = {target_distance:.2f})')
        logging.info(f'Speed: {speed:.2f}, acceleration: {acc:.2f}, waypoint angle: {target_angle:.2f}')

        wp_distances = np.array([x.distance_to(y) for (x, y) in pairwise([pos] + targets)])
        wp_cum_distances = wp_distances.cumsum()

        wp_angles = np.array([
            turn_angle(waypoint_angle(cur_pos, prev_pos, next_pos))
            for prev_pos, cur_pos, next_pos in
            zip([pos] + get_lines(), get_lines(), get_lines(1))
        ])

        wp_speeds = [max_corner_speed(a) for a in wp_angles]
        max_speeds = [max_speed(d, s) for d, s in zip(wp_cum_distances, wp_speeds)]

        rel_target = position.inverse() * next_target
        angle = rel_target.as_polar()[1]

        target_velocity = min(max_speeds)
        logging.info(f'Target velocity: {target_velocity:.2f}')
        if speed < target_velocity:
            throttle = 1
        else:
            logging.info('BRAKE!')
            throttle = -1

        self.prev_speed = speed
        self.prev_pos = pos
        if abs(angle) > 5:
            if angle > 0:
                return throttle, 1
            else:
                return throttle, -1
        else:
            return throttle, 0


def waypoint_angle(pos, prev_pos, next_pos) -> float:
    return (prev_pos - pos).angle_to(next_pos - pos)


def turn_angle(angle: float) -> float:
    """ 0 = straight line, 180 = reverse"""
    return abs(180 - abs(angle))


def max_speed(distance: float, desire_speed: float) -> float:
    return 60 * math.sqrt(
        (desire_speed / 60) ** 2 - 2 * MAX_DECELERATION / 60 * max(0.0, distance - 20)
    )


def max_corner_speed(angle: float) -> float:
    return float(np.interp(
        x=angle,
        xp=[0, 10, 15, 20, 50, 70, 90, 180],
        fp=[1000, 550, 500, 260, MIN_SPEED, MIN_SPEED, MIN_SPEED, MIN_SPEED]
    ))
