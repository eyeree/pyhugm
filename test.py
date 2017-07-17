from __future__ import division
from __future__ import print_function

import numpy as np


def interp_rgb(x, indexes, colors):
    r = np.interp(x, indexes, [c[0] for c in colors])
    g = np.interp(x, indexes, [c[1] for c in colors])
    b = np.interp(x, indexes, [c[2] for c in colors])
    return (r, g, b)


def iter_rgb(count, colors):
    indexes = [0, count - 1]
    for i in range(0, count):
        yield interp_rgb(i, indexes, colors)


foo = range(0, 32)
print('foo', foo)

foo[0:32] = iter_rgb(32, [(255, 180, 0), (0, 0, 0)])

print('foo', foo)





