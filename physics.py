# constants
import math

from pygame import Vector2

from linear_math import Rotation, Transform

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
    velocity = Rotation.fromangle(
        steering_command * max_steering_speed * dt * (1 - slipping_ratio)) * velocity

    # integrate acceleration
    delta_velocity = acceleration * dt
    velocity += delta_velocity

    # integrate velocity
    new_position = Transform(Rotation.fromangle(position.M.angle), position.p.copy())
    new_position.p += velocity * dt
    new_position.M *= Rotation.fromangle(steering_command * max_steering_speed * dt)

    return new_position

def max_angle(speed: float) -> float:
    position = Transform()
    velocity = Vector2.from_polar((speed, position.M.angle))

    new_position = update(position, velocity, 1, 1)

    #rad_turn_angle = position.M.angle - new_position.M.angle
    # turn_angle = rad_turn_angle / math.pi * 180
    rad_turn_angle = position.p.angle_to(new_position.p) / 180 * math.pi

    return rad_turn_angle

def max_speed(radius: float) -> float:
    steer_angle = .02
    turn_angle = math.pi - steer_angle

    speed = radius / math.tan(turn_angle / 2)
    return speed * 60

def radius_from_turn_angle(angle: float, track_width: float) -> float:
    a = angle / 180 * math.pi
    r = -.5 * track_width * math.sin(a / 2) / (math.sin(a / 2) - 1)
    return r + track_width / 2

if __name__ == '__main__':
    for r in [5, 10, 20, 50, 100]:
        print(f'Speed = {max_speed(r):.3f}: radius = {r}')

    print('Max angle')
    for speed in [1000, 500, 200, 100, 50]:
        print(f'Speed = {speed}: angle = {max_angle(speed)}')

    # print(radius_from_turn_angle(45, track_width=10))
