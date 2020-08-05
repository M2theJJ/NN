#compress

import bz2

#turn array into string
data = string

c = bz2.compress(data)
len(data) / len(c) #Data compression ratio

d = bz2.decompress(c)
data == d # Check equality to original object after round-trip