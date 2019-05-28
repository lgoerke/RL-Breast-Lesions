# Adapted from https://gist.github.com/TimSC/afda36eeb3dac249b589535f8a7ad7b5
import random
import math
import numpy as np

# https://gamedev.stackexchange.com/questions/23625/how-do-you-generate-tileable-perlin-noise

dirs = [(math.cos(a * 2.0 * math.pi / 256),
         math.sin(a * 2.0 * math.pi / 256))
        for a in range(256)]


def surflet(gridX, gridY, x, y, hashfunc):
    distX, distY = abs(x - gridX), abs(y - gridY)
    polyX = 1 - 6 * distX ** 5 + 15 * distX ** 4 - 10 * distX ** 3
    polyY = 1 - 6 * distY ** 5 + 15 * distY ** 4 - 10 * distY ** 3
    hashed = hashfunc(int(gridX), int(gridY))
    grad = (x - gridX) * dirs[hashed % len(dirs)][0] + (y - gridY) * dirs[hashed % len(dirs)][1]
    return polyX * polyY * grad


def noise(x, y, hashfunc):
    intX, intY = int(math.floor(x)), int(math.floor(y))
    s1 = surflet(intX + 0, intY + 0, x, y, hashfunc)
    s2 = surflet(intX + 1, intY + 0, x, y, hashfunc)
    s3 = surflet(intX + 0, intY + 1, x, y, hashfunc)
    s4 = surflet(intX + 1, intY + 1, x, y, hashfunc)
    return (s1 + s2 + s3 + s4)


def fBm(x, y, octs, hashfunc):
    val = 0
    for o in range(octs):
        scale = 2 ** o
        val += 0.5 ** o * noise(x * scale, y * scale, hashfunc)
    return val


class PermHash(object):
    def __init__(self, perm=None):
        if perm is None:
            self._perm = list(range(256))
            random.shuffle(self._perm)
            self._perm += self._perm
        else:
            self._perm = perm

    def __call__(self, *args):
        return self._perm[(self._perm[int(args[0]) % len(self._perm)] + int(args[1])) % len(self._perm)]

    def GetSaveState(self):
        return self._perm


def get_perlin_noise_img(height, width):
    freq = 1 / 32.0
    octs = 4
    data = []
    hashfunc = PermHash()

    for y in range(height):
        for x in range(width):
            tx = x - 100
            ty = y - 100
            data.append(fBm(tx * freq, ty * freq, octs, hashfunc))

    # one_channel = np.reshape(data, (height, width))*255
    one_channel = (np.array(data) - np.min(np.array(data))) / np.ptp(np.array(data))
    # one_channel = np.array(data)
    one_channel = np.reshape(one_channel, (height, width))*255
    # print(one_channel.shape)
    im = np.zeros((height, width, 3))
    im[:, :, 0] = one_channel
    im[:, :, 1] = one_channel
    im[:, :, 2] = one_channel
    return im.astype(int)
    # return im
