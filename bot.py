import logging
import math
import os
import pygame
import numpy as np

from typing import Tuple
from math import radians, degrees
from pygame import Vector2, Color, Surface
from itertools import cycle, islice, pairwise

from constants import framerate
from track import Track
from .physics import approx_max_corner_speed, radius_from_turn_angle
from ...bot import Bot
from ...linear_math import Transform

FRAMERATE = framerate
DT = 1 / FRAMERATE
MAX_ACCELERATION = 5.0 / 3.0
MAX_DECELERATION = -MAX_ACCELERATION
MIN_SPEED = 150
DEBUG = False

if os.getenv('DEBUG'):
    DEBUG = True
    logging.basicConfig(level='INFO')

class MinVerstappen(Bot):
    def __init__(self, track: Track):
        super().__init__(track)
        self.position = Transform()
        self.velocity = Vector2()
        self.prev_pos = Vector2()
        self.target = Vector2()
        self.prev_speed = 0


    @property
    def name(self):
        return "Min Verstappen"


    @property
    def contributor(self):
        return "Niek"


    @property
    def color(self) -> Color:
        return Color('#FF4F00')


    def compute_commands(self, next_waypoint: int, position: Transform, velocity: Vector2) -> Tuple:
        def get_lines(offset=0, limit=20):
            start = (next_waypoint + offset) % len(self.track.lines)
            return list(islice(cycle(self.track.lines), start, start + limit))


        pos = position.p
        speed = velocity.length()
        acc = speed - self.prev_speed  # t = 1
        target = self.track.lines[next_waypoint]
        targets = get_lines()
        target_distance = target.distance_to(pos)
        target_angle = turn_angle(waypoint_angle(target, pos, targets[1]))

        logging.info(f'NEXT WAYPOINT: {next_waypoint} (distance = {target_distance:.2f})')
        logging.info(
            f'Speed: {speed:.2f}, acceleration: {acc:.2f}, waypoint angle: {target_angle:.2f}')

        wp_distances = np.array([x.distance_to(y) for (x, y) in pairwise([pos] + targets)])
        wp_cum_distances = wp_distances.cumsum()

        wp_angles = np.array(
            [
                turn_angle(waypoint_angle(cur_pos, prev_pos, next_pos))
                for prev_pos, cur_pos, next_pos in
                zip([pos] + get_lines(), get_lines(), get_lines(1))
            ])

        wp_radii = [radius_from_turn_angle(radians(180 - a), self.track.track_width) for a in wp_angles]

        wp_speeds = [interp_max_corner_speed(a) for a in wp_angles]
        wp_speeds2 = [approx_max_corner_speed(r) for r in wp_radii]
        max_speeds = [max_entry_speed(d, s) for d, s in zip(wp_cum_distances, wp_speeds)]
        max_speeds2 = [max_entry_speed(d, s) for d, s in zip(wp_cum_distances, wp_speeds2)]

        rel_target = position.inverse() * target
        angle = rel_target.as_polar()[1]

        target_velocity = 20 + min(max_speeds2)
        logging.info(f'Target velocity: {target_velocity:.2f}, alt = {min(max_speeds2):.2f}')
        if speed < target_velocity:
            throttle = 1
        else:
            logging.info('BRAKE!')
            throttle = -1

        self.prev_speed = speed
        self.prev_pos = pos
        self.target = target
        self.position = position
        self.velocity = velocity

        if abs(angle) > 5:
            if angle > 0:
                return throttle, 1
            else:
                return throttle, -1
        else:
            return throttle, 0


    def draw(self, map_scaled: Surface, zoom: float):
        if not DEBUG:
            return

        sideways_velocity = (self.velocity * self.position.M.cols[1]) * self.position.M.cols[1]

        # path = [self.prev_pos.lerp(self.target, t) * zoom for t in np.linspace(0, 1, num=10)]
        pygame.draw.line(
            map_scaled,
            start_pos=self.prev_pos * zoom, end_pos=(self.prev_pos + sideways_velocity) * zoom,
            color=(0, 0, 0), width=2)

        # pygame.draw.lines(map_scaled, points=path, closed=False, color=(0, 0, 0))


def waypoint_angle(pos, prev_pos, next_pos) -> float:
    return (prev_pos - pos).angle_to(next_pos - pos)


def turn_angle(angle: float) -> float:
    """ 0 = straight line, 180 = reverse"""
    return abs(180 - abs(angle))


def max_entry_speed(distance: float, desire_speed: float) -> float:
    return 60 * math.sqrt(
        (desire_speed / 60) ** 2 - 2 * MAX_DECELERATION / 60 * max(0.0, distance - 20)
    )


def interp_max_corner_speed(angle: float) -> float:
    return float(
        np.interp(
            x=angle,
            xp=[0, 10, 15, 20, 50, 70, 90, 180],
            fp=[1000, 550, 500, 260, MIN_SPEED, MIN_SPEED, MIN_SPEED, MIN_SPEED]
        ))
