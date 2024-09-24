import functools
from math import isclose, pi, radians, sin, tan

import numpy as np
from pygame import Vector2
from scipy.optimize import minimize

from .util import assert_radians, copy_rot, eval_1d
from ...linear_math import Rotation, Transform

# constants
max_throttle = 100
max_steering_speed = 3
slipping_acceleration = 200
slipping_ratio = 0.6
dt = 1.0 / 60

def update(position: Transform, velocity: Vector2, throttle: float, steering_command: float):
    acceleration = position.M * Vector2(throttle * max_throttle, 0)

    sideways_velocity = (velocity * position.M.cols[1]) * position.M.cols[1]

    if sideways_velocity.length_squared() > 0.001:
        # slow down the car in sideways direction
        acceleration -= sideways_velocity.normalize() * slipping_acceleration

    # rotate velocity partially
    # steering_angle = .02
    steering_angle = steering_command * max_steering_speed * dt * (1 - slipping_ratio)
    velocity = Rotation.fromangle(steering_angle) * velocity

    # integrate acceleration
    delta_velocity = acceleration * dt
    velocity += delta_velocity

    # integrate velocity
    new_position = Transform(Rotation.fromangle(position.M.angle), position.p.copy())
    new_position.p += velocity * dt
    new_position.M *= Rotation.fromangle(steering_command * max_steering_speed * dt)

    return new_position, velocity


def delta_position(velocity: Vector2, rot: Rotation = None) -> Vector2:
    position = Transform(M=copy_rot(rot), p=Vector2(0, 0))
    new_position, new_velocity = update(position, velocity, 1, 1)
    return new_position


def update_velocity(velocity: Vector2, rot: Rotation = None) -> Vector2:
    position = Transform(M=copy_rot(rot), p=Vector2(0, 0))
    new_position, new_velocity = update(position, velocity, 1, 1)
    return new_velocity


def max_turning_angle(speed: float, drift_angle: float) -> float:
    assert_radians(drift_angle)

    velocity = Vector2()
    velocity.from_polar((speed, 0))
    rot = Rotation.fromangle(-drift_angle)
    new_velocity = update_velocity(velocity, rot)
    angle = radians(velocity.angle_to(new_velocity))
    return angle


def approx_max_corner_speed(radius: float, steer_angle: float = .02) -> float:
    assert_radians(steer_angle)

    corner_angle = pi - steer_angle
    speed = radius / tan(corner_angle / 2)
    return speed * 60


@functools.lru_cache()
def max_corner_speed(radius: float) -> float:
    def objective(speed):
        velocity = Vector2.from_polar((speed[0], 0))
        pos = Vector2()
        prev_pos = pos - velocity / 60
        rot = Rotation.fromangle(-best_drift_angle(round(speed[0])))
        new_pos = delta_position(velocity, rot)
        actual_radius = compute_local_turning_radius(prev_pos, pos, new_pos.p)
        return (actual_radius - radius) ** 2

    result = minimize(objective, np.array(300), bounds=((100.0, 750.0),))
    return float(result.x[0])

@functools.lru_cache()
def max_corner_drift_speed(radius: float) -> float:
    """Max corner speed under optimal drift angle"""
    def objective(speed):
        speed = float(speed)
        corner_angle = pi - max_turning_angle(speed, best_drift_angle(speed))
        actual_radius = speed / 60 * tan(corner_angle / 2)
        return (actual_radius - radius) ** 2

    result = minimize(objective, np.array(150), bounds=((100.0, 999.0),))
    return float(result.x[0])


# def max_corner_speed_for_drift(radius: float, drift_angle: float) -> float:
#     assert_radians(drift_angle)
#
#     steer_angle = max_turning_angle(150, drift_angle=drift_angle)
#     return max_corner_speed(radius, steer_angle)


@functools.lru_cache()
def best_drift_angle(speed: float) -> float:
    assert 0 < speed < 1000
    result = minimize(lambda x: -max_turning_angle(speed, x), np.array(.49))
    return float(result.x[0])


def radius_from_turn_angle(angle: float, track_width: float) -> float:
    assert_radians(angle)
    assert track_width > 0

    a = angle / 2
    assert not isclose(sin(a), 1), f'invalid angle {angle}'

    r = -.5 * track_width * sin(a) / (sin(a) - 1)
    return r + track_width / 2


def compute_turning_radius(positions):
    dx = np.diff(positions[:, 0])
    dy = np.diff(positions[:, 1])
    ds = np.sqrt(dx ** 2 + dy ** 2)

    angles = np.abs(np.arctan2(dy, dx))
    dtheta = np.diff(angles)

    curvature = np.abs(dtheta) / ds[:-1]
    turning_radius = 1 / curvature

    assert len(turning_radius) == 1
    return float(turning_radius[0])


def compute_local_turning_radius(prev_pos: Vector2, pos: Vector2, next_pos: Vector2) -> float:
    positions = np.array([tuple(prev_pos), tuple(pos), tuple(next_pos)])
    return compute_turning_radius(positions)


if __name__ == '__main__':
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    pd.options.display.width = None
    pd.options.display.max_rows = None
    # for r in [5, 10, 20, 50, 100]:
    #     print(f'Speed = {max_speed(r):.3f}: radius = {r}')
    #
    # print('Max angle under drifting')
    # for drift_angle in [0, 5, 10, 28.35, 45, 90, 135, 180]:
    #     print(f'Drift angle = {drift_angle}: turn angle = {max_turning_angle(speed=500, drift_angle=drift_angle)}')
    # exit()

    speed_data2 = eval_1d(max_corner_speed2, np.linspace(50, 1000, 100), 'radius', 'speed')
    speed_data1 = eval_1d(max_corner_speed, np.linspace(50, 1000, 100), 'radius', 'speed')
    speed_data2['method'] = '2'
    speed_data1['method'] = '1'

    speed_data = pd.concat([speed_data1, speed_data2], ignore_index=True)

    sns.lineplot(speed_data, x='radius', y='speed', hue='method')
    plt.show()

    exit()



    r = radius_from_turn_angle(radians(180 - 20), 60)
    print('RADIUS: ', r)
    print(max_corner_drift_speed(r))
    exit()

    print('Max angle, without drifting')
    for speed in [1000, 500, 200, 100, 50]:
        print(f'Speed = {speed}: turn angle = {max_turning_angle(speed=150)}')

    # print(radius_from_turn_angle(45, track_width=10))

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    sns.set()

    # max angle from speed (no drifting)
    angle_data = eval_1d(
        max_turning_angle, np.linspace(0, 1000, 1000), 'speed', 'angle')
    # sns.lineplot(angle_data, x='speed', y='angle')
    # plt.show()

    # max angle from drift angle (for given speed)
    angle_drift_data = eval_1d(
        lambda x: max_turning_angle(speed=500, drift_angle=x), np.linspace(0, radians(360), 1000),
        'drift_angle', 'angle')
    print(angle_drift_data)

    print(angle_drift_data.nlargest(n=1, columns='angle'))

    sns.lineplot(angle_drift_data, x='drift_angle', y='angle')
    plt.show()

    # max speed from radius
    speed_data = eval_1d(
        max_corner_speed, np.linspace(0, 100, 1000), 'radius', 'speed')
    # sns.lineplot(speed_data, x='radius', y='speed')

    # radius from turn angle
    turn_data = eval_1d(
        partial(radius_from_turn_angle, track_width=30),
        np.linspace(0, radians(360), 1000), 'angle', 'radius')
    # ax = sns.lineplot(turn_data, x='angle', y='radius')
    # ax.set(ylim=(00, 1000))

    # radius from turn angle with track width
    turn_width_data = eval_2d(
        radius_from_turn_angle,
        list(map(radians, range(5, 360, 10))),
        list(map(radians, range(5, 50, 2))),
        'angle', 'track_width', 'radius'
    )

    # sns.heatmap(turn_width_data[turn_width_data['radius'] < 500].pivot(index='track_width', columns='angle', values='radius'), annot=False, cmap='viridis')

    # max angle from speed and drift angle
    angle_drift_data = eval_2d(
        max_turning_angle,
        np.linspace(100, 1000, 10),
        list(map(radians, range(0, 360, 10))),
        'speed', 'drift_angle', 'angle')

    # sns.heatmap(angle_drift_data.pivot(index='speed', columns='drift_angle', values='angle'), annot=False, cmap='viridis')

    plt.show()
