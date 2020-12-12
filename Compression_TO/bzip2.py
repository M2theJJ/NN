import bz2

import numpy as np
import binascii

def bzip2_Do(array):
#string = '1Ea5'
#arr = np.random.randint(1, 101, size=(5, 5))

    data = bytes(array)
    print('this is the list', data)

    c = bz2.compress(data)
    print('compression ratio:', len(c)/len(data)) #Data compression ratio
    print('compressed', c)

    d = bz2.decompress(c)
    assert data == d # Check equality to original object after round-trip
    print('decompressed', d)

    return c, d

#works properly so far