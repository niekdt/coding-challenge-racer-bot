import numpy as np

from math import pi
from typing import Callable

from ...linear_math import Rotation


QUOTES = [
    "Je m'appele frikandel!",
    'Simply lovely',
    'What is he doing?!',
    'That was stupid',
    'I have no grip!',
    'Unbelievable!',
    'This strategy makes no sense!',
    'The car is undrivable!',
    "The thing just doesn't f*** turn",
    'f***',
    'LOL',
    'This is a joke, mate!',
    'Mate, this is ridiculous!',
    'What a joke',
    "I'm not here to finish fourth",
    'This is just stupid!',
    'This is a f*** disaster',
    'This is a mess',
    'I am losing power',
    'Why do these guys not move?',
    'Traffic is a joke',
    'This is just ridiculous',
    'What a f*** idiot',
    'What the f*** happened here?',
    'My tire is gone!',
    'This car is on rails!'
]


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


def assert_radians(angle: float):
    assert abs(angle) <= 2 * pi, 'use radians!'
