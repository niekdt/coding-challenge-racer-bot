import logging
import math
import os
from itertools import cycle, islice, pairwise
from typing import Tuple

import numpy as np
import pygame
from pygame import Color, Surface, Vector2
from scipy.interpolate import splev, splprep

from ...constants import framerate
from ...track import Track
from .physics import compute_local_turning_radius, max_corner_speed
from ...bot import Bot
from ...linear_math import Transform

DT = 1 / framerate
MAX_ACCELERATION = 100
MAX_DECELERATION = -110
DEBUG = False

if os.getenv('DEBUG'):
    DEBUG = True
    logging.basicConfig(level='INFO')
    np.set_printoptions(precision=2)

class MinVerstappen(Bot):
    def __init__(self, track: Track):
        super().__init__(track)
        self.position = Transform()
        self.velocity = Vector2()

        self.target = Vector2()
        self.prev_pos = Vector2()
        self.prev_speed = 0
        self.prev_target_speed = 0
        self.prev_velocity = Vector2()

        if track:
            self.nodes, self.path = fit_racingline(self.track.lines, int(1e3))

        self.last_checkpoint = 0
        self.target_checkpoint = 1

        if DEBUG:
            self.font = pygame.font.Font(pygame.font.get_default_font(), 10)
            self.font2 = pygame.font.Font(pygame.font.get_default_font(), 14)


    @property
    def name(self):
        return "Min Verstappen"


    @property
    def contributor(self):
        return "Niek"


    @property
    def color(self) -> Color:
        return Color('#FF4F00')


    def get_nodes(self, last_checkpoint, offset=0, limit=10):
        start = (last_checkpoint + offset) % len(self.nodes)
        return list(islice(cycle(self.nodes), start, start + limit))


    def compute_commands(self, next_waypoint: int, position: Transform, velocity: Vector2) -> Tuple:
        logging.info(f'\n== New turn == (speed = {velocity.length():.0f})')
        pos = position.p
        next_neutral_pos = pos + velocity / 60
        speed = velocity.length()

        front_nodes = self.nodes[self.last_checkpoint:(self.last_checkpoint + 10)]
        front_path = self.path[self.last_checkpoint:(self.last_checkpoint + 10)]
        assert front_nodes.shape[0], 'no next nodes!'

        # check which segment we're in
        distances = np.linalg.norm(
            front_nodes - np.array([next_neutral_pos[0], next_neutral_pos[1]]), axis=1)
        nearest_index = np.argmin(distances)
        assert nearest_index >= 0

        if nearest_index > 0:
            logging.info(f'Approaching new checkpoint {self.last_checkpoint + nearest_index}')
            # check if we've crossed into a next segment
            start_diff = front_path[nearest_index] - pos
            segment_diff = front_path[nearest_index] - front_path[nearest_index + 1]
            in_segment = 0 < start_diff.dot(segment_diff) / segment_diff.dot(segment_diff) < 1

            if in_segment:
                logging.info('Entered next segment!')
                self.last_checkpoint = self.last_checkpoint + nearest_index
            else:
                logging.info('Entered new segment!')
                self.last_checkpoint = self.last_checkpoint + nearest_index - 1

        # set target checkpoint
        self.target_checkpoint = self.last_checkpoint + 1
        target = self.path[self.target_checkpoint]
        target_distance = pos.distance_to(target)
        logging.info(f'Target checkpoint: {self.target_checkpoint} (distance = {target_distance:.2f}')

        targets = self.path[self.target_checkpoint:self.target_checkpoint + 100]

        wp_distances = np.array([x.distance_to(y) for (x, y) in pairwise([pos] + targets)])
        wp_cum_distances = wp_distances.cumsum()

        wp_radii = np.array([
            compute_local_turning_radius(prev_pos, cur_pos, next_pos)
            for prev_pos, cur_pos, next_pos in
            zip(targets[0:], targets[1:], targets[2:])
        ])
        logging.info(f'radii = {wp_radii[1:10]}')

        wp_speeds = [max_corner_speed(round(r)) for r in wp_radii]
        wp_speeds = np.convolve(wp_speeds, np.ones(5) / 5, mode='valid')
        logging.info(f'speeds = {wp_speeds[1:10]}')
        max_speeds = np.asarray([max_entry_speed(d, s) for d, s in zip(wp_cum_distances, wp_speeds)])

        target_speed = max_speeds.min()
        limit_checkpoint = max_speeds.argmin()
        logging.info(f'Target velocity: {target_speed:.2f}, limited by {limit_checkpoint}th checkpoint ahead')
        if speed < target_speed:
            throttle = 1
        else:
            logging.info('BRAKE!')
            throttle = -1
        self.prev_speed = speed
        self.prev_target_speed = target_speed
        self.prev_pos = pos
        self.target = target
        self.position = position
        self.velocity = velocity

        # determine steering angle
        angle_target = self.path[self.target_checkpoint + 12]
        rel_target = position.inverse() * angle_target
        angle = rel_target.as_polar()[1]
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

        pygame.draw.line(
            map_scaled,
            start_pos=self.prev_pos * zoom, end_pos=(self.prev_pos + sideways_velocity) * zoom,
            color=(0, 0, 0), width=2)

        # draw speed
        text_surface = self.font2.render(f'{self.prev_speed:.0f}', True, self.color)
        map_scaled.blit(text_surface, dest=self.position.p * zoom - Vector2(25, 75))

        # draw target speed
        text_surface = self.font2.render(
            f'{self.prev_target_speed:.0f}', True,
            '#FF0000' if self.prev_speed > self.prev_target_speed else '#00FF00')
        map_scaled.blit(text_surface, dest=self.position.p * zoom - Vector2(-25, 75))

        for i, p in enumerate(self.path[:-5]):
            pygame.draw.circle(map_scaled, center=p * zoom, radius=1, color='#000000', width=50)
        #     if i % 3 == 0:
        #         text_surface = self.font.render(f'{self.info[i]:.1f}', True, '#FFFFFF')
        #         map_scaled.blit(text_surface, dest=p * zoom)
        # pygame.draw.circle(map_scaled, center=self.path[self.last_checkpoint] * zoom, radius=10, color='#FFFFFF', width=2)
        pygame.draw.circle(
            map_scaled, center=self.path[self.target_checkpoint] * zoom,
            radius=10, color='#FF0000', width=3)


def max_entry_speed(distance: float, desire_speed: float) -> float:
    return math.sqrt(
        desire_speed ** 2 - 2 * MAX_DECELERATION * max(0.0, distance - 0)
    )


def fit_racingline(waypoints: list[Vector2], res) -> tuple[list[Vector2], np.ndarray]:
    x, y = fit_polycurve(waypoints + waypoints[0:3], n=res)

    # window = 3
    # x = np.convolve(x, np.ones(window) / window, mode='valid')
    # y = np.convolve(y, np.ones(window) / window, mode='valid')

    coords = np.column_stack([x, y])
    path = [Vector2(float(x), float(y)) for x, y in zip(x, y)]

    return coords, path


def fit_polycurve(positions: list[Vector2], n: int) -> tuple[float, float]:
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]

    tck, u = splprep([x, y], s=0, k=2)

    u_new = np.linspace(0, 1, n)
    x_new, y_new = splev(u_new, tck)
    return x_new, y_new


def circumcircle_radius(prev_pos: Vector2, pos: Vector2, next_pos: Vector2) -> float:
    a = pos.distance_to(next_pos)
    b = prev_pos.distance_to(next_pos)
    c = prev_pos.distance_to(pos)

    # Calculate the area using the determinant method
    K = .5 * np.abs(
        prev_pos[0] * (pos[1] - next_pos[1]) + pos[0] * (next_pos[1] - prev_pos[1]) +
        next_pos[0] * (prev_pos[1] - pos[1])
    )

    # Calculate the circumcircle radius
    R = (a * b * c) / (4 * K)
    return R
