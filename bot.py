import logging
import os
import random
from itertools import pairwise
from typing import Tuple

import numpy as np
import pygame
from pygame import Color, Surface, Vector2

from .pathing import BsPath
from .util import QUOTES
from ...constants import framerate
from ...track import Track
from .physics import compute_local_turning_radius, max_corner_speed, max_entry_speed
from ...bot import Bot
from ...linear_math import Transform

N_ROUNDS = 3
DT = 1 / framerate
MAX_ACCELERATION = 100
MAX_DECELERATION = -110
DEBUG = False
RES = 2850

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
        self.quote = ''

        if track:
            self.path = BsPath(waypoints=self.track.lines, n_rounds=N_ROUNDS, res=RES)

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

    def compute_commands(self, next_waypoint: int, position: Transform, velocity: Vector2) -> Tuple:
        self.position = position
        self.velocity = velocity

        logging.info(f'\n== New turn == (speed = {velocity.length():.0f})')
        pos = position.p
        speed = velocity.length()

        self.last_checkpoint = self.path.get_last_checkpoint(
            pos, velocity, cp=self.last_checkpoint, limit=10)
        self.target_checkpoint = self.last_checkpoint + 1

        target = self.path.get_node(self.target_checkpoint)
        target_distance = pos.distance_to(target)
        logging.info(f'Target checkpoint: {self.target_checkpoint} (distance = {target_distance:.2f}')

        targets = self.path.get_nodes(cp=self.target_checkpoint, limit=100)

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
        max_speeds = np.asarray([
            max_entry_speed(d, s, MAX_DECELERATION)
            for d, s in zip(wp_cum_distances, wp_speeds)
        ])

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

        # determine steering angle
        angle_target = self.path.get_node(cp=self.target_checkpoint + 12)
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

        for i, p in enumerate(self.path.get_nodes(cp=self.last_checkpoint, limit=100)):
            pygame.draw.circle(map_scaled, center=p * zoom, radius=1, color='#000000', width=50)
        #     if i % 3 == 0:
        #         text_surface = self.font.render(f'{self.info[i]:.1f}', True, '#FFFFFF')
        #         map_scaled.blit(text_surface, dest=p * zoom)
        # pygame.draw.circle(map_scaled, center=self.nodes[self.last_checkpoint] * zoom, radius=10, color='#FFFFFF', width=2)
        pygame.draw.circle(
            map_scaled, center=self.path.get_node(self.target_checkpoint) * zoom,
            radius=10, color='#FF0000', width=3)

        if random.randrange(5 * framerate) == 1:
            if self.quote:
                self.quote = ''
            else:
                self.quote = random.choice(QUOTES)
        text_surface = self.font.render(self.quote, True, '#FFFFFF')
        map_scaled.blit(text_surface, dest=self.position.p * zoom - Vector2(text_surface.get_width() / 2, 30))
