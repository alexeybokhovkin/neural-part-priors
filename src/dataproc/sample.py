import os, sys
import struct
import numpy as np


class Sample:
    def __init__(self, dims=[0, 0, 0], res=0, grid2world=None, num_locations=None, locations=None, sdfs=None):
        self.filename = ""
        self.dimx = dims[0]
        self.dimy = dims[1]
        self.dimz = dims[2]
        self.res = res
        self.grid2world = grid2world
        self.num_locations = num_locations
        self.locations = locations
        self.sdfs = sdfs


def load_sample(filename):
    assert os.path.isfile(filename), "file not found: %s" % filename
    if filename.endswith(".df"):
        f_or_c = "C"
    else:
        f_or_c = "F"

    fin = open(filename, 'rb')

    s = Sample()
    s.filename = filename
    s.dimx = struct.unpack('Q', fin.read(8))[0]
    s.dimy = struct.unpack('Q', fin.read(8))[0]
    s.dimz = struct.unpack('Q', fin.read(8))[0]
    s.res = struct.unpack('f', fin.read(4))[0]
    n_elems = s.dimx * s.dimy * s.dimz

    s.grid2world = struct.unpack('f' * 16, fin.read(16 * 4))

    s.num_locations = struct.unpack('Q', fin.read(8))[0]
    try:
        location_bytes = fin.read(s.num_locations * 3 * 4)
        s.locations = struct.unpack('I' * 3 * s.num_locations, location_bytes)

        sdfs_bytes = fin.read(s.num_locations * 4)
        s.sdfs = struct.unpack('f' * s.num_locations, sdfs_bytes)
    except struct.error:
        print("Cannot load", filename)

    fin.close()
    s.grid2world = np.asarray(s.grid2world, dtype=np.float32).reshape([4, 4], order=f_or_c)
    s.locations = np.asarray(s.locations, dtype=np.uint32).reshape([s.num_locations, 3], order="C")
    s.sdfs = np.array(s.sdfs)

    return s