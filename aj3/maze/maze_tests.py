import unittest

import numpy as np

from aj3.maze.generator import DungeonRooms


def generate_dungeon():
    d = DungeonRooms(5, 5)
    out = d.generate()
    np.save('../data/maze.npy', out)


if __name__ == '__main__':
    generate_dungeon()
