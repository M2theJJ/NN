import bz2

import numpy as np
import binascii


#string = '1Ea5'
arr = np.random.randint(1, 101, size=(5, 5))

data = bytes(arr)
print('this is the list', data)

c = bz2.compress(data)
len(data) / len(c) #Data compression ratio
print('compressed', c)

d = bz2.decompress(c)
data == d # Check equality to original object after round-trip
print('decompressed', d)

#works properly so far