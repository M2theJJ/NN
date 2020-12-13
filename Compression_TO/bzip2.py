import bz2

import numpy as np
import binascii
import sys

def bzip2_Do(array):
    print('BZIP2:')
#string = '1Ea5'
#arr = np.random.randint(1, 101, size=(5, 5))

    data = bytes(array)
    d_s = sys.getsizeof(data)
    print('this is the list', data)

    c = bz2.compress(data)
    c_s = sys.getsizeof(c)
    print('compressed', c)

    d = bz2.decompress(c)
    assert data == d # Check equality to original object after round-trip
    print('decompressed', d)

    print('compression ratio:', len(c)/len(data)) #Data compression ratio
    print('compression ratio in bytes', c_s/d_s)

    return c, d

#works properly so far