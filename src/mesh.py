import jax
import numpy as np
from jax._src.mesh import Mesh


def get_device_grid():
    by_proc = {}
    for d in jax.devices():
        by_proc.setdefault(d.process_index, []).append(d)
    hosts = sorted(by_proc)  # stable host order
    # dev_grid[x, y] = device with local index x on host y
    return np.array([[by_proc[h][x] for h in hosts] for x in range(jax.local_device_count())]).T


def create_2d_mesh():
    dev_grid = get_device_grid()
    return Mesh(dev_grid, ("S", "T"))


def create_1d_mesh():
    dev_grid = get_device_grid()
    return Mesh(dev_grid.flatten(), ("S",))