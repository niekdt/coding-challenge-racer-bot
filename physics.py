from functools import partial
from math import pi, sin, tan, radians, isclose

from pygame import Vector2

from linear_math import Rotation, Transform

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


def delta_velocity(velocity: Vector2, rot: Rotation = None) -> Vector2:
    position = Transform(M=copy_rot(rot), p=Vector2(0, 0))
    new_position, new_velocity = update(position, velocity, 1, 1)
    return new_velocity


def max_turning_angle_vec(velocity: Vector2, rot: Rotation = None) -> float:
    position = Transform(M=copy_rot(rot), p=Vector2(0, 0))

    new_position, new_velocity = update(position, velocity, 1, 1)

    angle_p = radians(position.p.angle_to(new_position.p))
    angle_v = radians(velocity.angle_to(new_velocity))

    print(f'Angle P: {angle_p:.3f}, Angle V: {angle_v:.3f}')

    return angle_v #, new_velocity.length() - velocity.length(), side_velocity.length()


def max_turning_angle(speed: float, drift_angle: float = 0) -> float:
    velocity = Vector2.from_polar((speed, 0))
    rot = Rotation.fromangle(radians(-drift_angle))
    return max_turning_angle_vec(velocity, rot)


def max_speed(radius: float, steer_angle: float = .02) -> float:
    turn_angle = pi - steer_angle
    speed = radius / tan(turn_angle / 2)
    return speed * 60


def radius_from_turn_angle(angle: float, track_width: float) -> float:
    a = radians(angle) / 2
    assert not isclose(sin(a), 1), f'invalid angle {angle}'

    r = -.5 * track_width * sin(a) / (sin(a) - 1)
    return r + track_width / 2


from typing import Callable

import numpy as np

from linear_math import Rotation

def eval_1d(fun: Callable, x_values, x_label: str = 'x', y_label: str = 'y'):
    import pandas as pd
    y_values = [fun(x) for x in x_values]
    return pd.concat([
        pd.Series(list(x_values), name=x_label),
        pd.Series(y_values, name=y_label)
    ], axis=1)


def eval_2d(
    fun: Callable,
    x_values, y_values,
    x_label: str = 'x', y_label: str = 'y', z_label: str = 'z'
):
    import pandas as pd
    xx, yy = np.meshgrid(x_values, y_values)
    f = np.vectorize(fun)
    zz = f(xx, yy)

    df = pd.DataFrame({
        'x': xx.flatten(),
        'y': yy.flatten(),
        'z': zz.flatten()
    })

    return df.rename(columns={'x': x_label, 'y': y_label, 'z': z_label})


def copy_rot(rot: Rotation) -> Rotation:
    if rot is None:
        return Rotation.fromangle(0)
    else:
        return Rotation(rot.rows[0].x, rot.rows[1].x, rot.rows[0].y, rot.rows[1].y)



if __name__ == '__main__':
    import pandas as pd
    pd.options.display.width = None
    pd.options.display.max_rows = None
    # for r in [5, 10, 20, 50, 100]:
    #     print(f'Speed = {max_speed(r):.3f}: radius = {r}')
    #
    # print('Max angle under drifting')
    # for drift_angle in [0, 5, 10, 28.35, 45, 90, 135, 180]:
    #     print(f'Drift angle = {drift_angle}: turn angle = {max_turning_angle(speed=500, drift_angle=drift_angle)}')
    # exit()

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
        lambda x: max_turning_angle(speed=500, drift_angle=x), np.linspace(0, 180, 1000), 'drift_angle', 'angle')
    print(angle_drift_data)

    print(angle_drift_data.nlargest(n=1, columns='angle'))

    sns.lineplot(angle_drift_data, x='drift_angle', y='angle')
    plt.show()


    # max speed from radius
    speed_data = eval_1d(
        max_speed, np.linspace(0, 100, 1000), 'radius', 'speed')
    # sns.lineplot(speed_data, x='radius', y='speed')

    # radius from turn angle
    turn_data = eval_1d(
        partial(radius_from_turn_angle, track_width=30),
        np.linspace(0, 360, 1000), 'angle', 'radius')
    # ax = sns.lineplot(turn_data, x='angle', y='radius')
    # ax.set(ylim=(00, 1000))

    # radius from turn angle with track width
    turn_width_data = eval_2d(
        radius_from_turn_angle,
        range(5, 360, 10),
        range(5, 50, 2),
        'angle', 'track_width', 'radius'
    )

    # sns.heatmap(turn_width_data[turn_width_data['radius'] < 500].pivot(index='track_width', columns='angle', values='radius'), annot=False, cmap='viridis')

    # max angle from speed and drift angle
    angle_drift_data = eval_2d(
        max_turning_angle,
        np.linspace(100, 1000, 10),
        range(0, 360, 10),
        'speed', 'drift_angle','angle')

    # sns.heatmap(angle_drift_data.pivot(index='speed', columns='drift_angle', values='angle'), annot=False, cmap='viridis')

    plt.show()
