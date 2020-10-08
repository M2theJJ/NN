import numpy as np
import struct
from Extras import RAS

array = np.random.randint(1, 3, size=(3, 3))
#array = array.flatten()
print('array', array)
list = RAS.convert_array_to_list(array)
c_array = [list.count(x) for x in set(list)]
print('c_array', c_array)

def Streaming_rANS_encoder(s_input, symbol_counts, range_factor):
  total_counts = np.sum(symbol_counts)  # Represents M
  bitstream = [] #initialize stream
  state = low_level*total_counts #state initialized to lM

  for s in s_input: #iterate over the input
    # Output bits to the stream to bring the state in the range for the next encoding
    while state >= range_factor*symbol_counts[s]:
      bitstream.append( state%2 )
      state = state/2

    state = C_rANS(s, state, symbol_counts) # The rANS encoding step
  return state, bitstream