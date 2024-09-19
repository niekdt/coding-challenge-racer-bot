import logging
import math

import numpy as np

from typing import Tuple
from pygame import Vector2
from itertools import cycle, islice, pairwise

from track import Track
from ...bot import Bot
from ...linear_math import Transform

MAX_ACCELERATION = 5.0 / 3.0
MAX_DECELERATION = -MAX_ACCELERATION
MAX_CORNER_SPEED = 130


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
        logging.info(f'NEXT WAYPOINT: {next_waypoint}')
        pos = position.p

        speed = velocity.length()
        acc = speed - self.prev_speed  # t = 1
        next_target = self.track.lines[next_waypoint]
        target_distance = next_target.distance_to(pos)
        logging.info(f'Speed: {speed:.2f}, acceleration: {acc:.2f}')

        def get_lines(offset=0, limit=20):
            start = (next_waypoint + offset) % len(self.track.lines)
            return list(islice(cycle(self.track.lines), start, start + limit))

        targets = get_lines()
        wp_distances = np.array([x.distance_to(y) for (x, y) in pairwise([pos] + targets)])

        wp_angles = np.array([
            turn_angle((prev_wp_pos - cur_wp_pos).angle_to(next_wp_pos - cur_wp_pos))
            for prev_wp_pos, cur_wp_pos, next_wp_pos in
            zip([pos] + get_lines(), get_lines(), get_lines(1))
        ])

        sharp_wp = np.where(abs(wp_angles) > 40)[0][0]
        sharp_wp_distance = wp_distances[:(sharp_wp + 1)].sum()
        logging.info(f'Sharp turn is {sharp_wp} waypoints away (distance {sharp_wp_distance:.2f})')

        rel_target = position.inverse() * next_target
        angle = rel_target.as_polar()[1]

        target_velocity = math.sqrt((MAX_CORNER_SPEED / 60) ** 2 - 2 * MAX_DECELERATION / 60 * sharp_wp_distance) * 60
        logging.info(f'Target_velocity: {target_velocity:.2f}')
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


def turn_angle(angle: float) -> float:
    """ 0 = straight line, 180 = reverse"""
    return abs(180 - angle)
