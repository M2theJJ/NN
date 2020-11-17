import numpy as np
from Extras import RAS
import random

arr = np.random.randint(1, 8, size=(3, 3))
arr = RAS.convert_array_to_list(arr)
arr = list(map(int, arr))
print('array', arr)

RANS64_L = 2**30
MIN_PROB = 8
prob_bits = 14
prob_scale = 1 << prob_bits

def argmax(values):
    if not values:
        return -1 # Empty list has no argmax

    current_max = values[0]
    current_max_index = 0
    for i in range(1,len(values)):
        if values[i] > current_max:
            current_max = values[i]
            current_max_index = i

    return current_max_index



def float_to_int_probs(float_probs):
    pdf = []
    cdf = [0]

    for prob in float_probs:
        next_prob = round(prob * prob_scale)
        if prob > 0 and next_prob < MIN_PROB:
            next_prob = MIN_PROB

        pdf.append(next_prob)
        cdf.append(cdf[-1] + next_prob)

    # Account for possible rounding error
    # Remove the correction from the largest element
    to_correct = prob_scale - cdf[-1]

    largest_index = argmax(pdf)
    pdf[largest_index] += to_correct
    for i in range(largest_index + 1, len(cdf)):
        cdf[i] += to_correct


    return (pdf, cdf)

def find_in_int_dist(cdf, to_find):

    for i in range(len(cdf) - 1):
        if cdf[i] <= to_find and cdf[i + 1] > to_find:
            return i

    print ("Error: Could not find symbol in integer-dist")

class Encoder:
    def __init__(self):
        self.state = RANS64_L
        self.encoded_data = []

    def encode_symbol(self, freqs, symbol):
        (pdf, cdf) =  float_to_int_probs(freqs)
        print('pdf', pdf, 'cdf', cdf, 'symbol', symbol)
        freq = pdf[symbol]
        start = cdf[symbol]

        if freq == 0:
            print ("Error: Can't encode symbol with frequency 0!")
            return

        x = self.state

        x_max = ((RANS64_L >> prob_bits) << 32) * freq
        if x >= x_max:
            self.encoded_data.append(x & 0xffffffff)
            x >>= 32

        self.state = ((x // freq) << prob_bits) + (x % freq) + start

    def get_encoded(self):
        self.encoded_data.append(self.state & 0xffffffff)
        self.state >>= 32
        self.encoded_data.append(self.state & 0xffffffff)
        print('encoded data', self.encoded_data)
        return self.encoded_data

class Decoder:
    def __init__(self, encoded_data):
        self.state = (encoded_data.pop() << 32) | encoded_data.pop()
        self.encoded_data = encoded_data

    def decode_symbol(self, freqs):
        (pdf, cdf) = float_to_int_probs(freqs)

        # Extract symbol
        to_find = self.state & (prob_scale - 1)
        symbol = find_in_int_dist(cdf, to_find)

        # Symbol related variables.
        start = cdf[symbol]
        freq = pdf[symbol]

        mask = prob_scale - 1

        # Move state foward one step
        x = self.state
        x = freq * (x >> prob_bits) + (x & mask) - start
        print('x', x)

        # Enough of state has been read that we now need to get more out of encoded_data.
#        if x < RANS64_L:
#            x = (x << 32) | self.encoded_data.pop()

        self.state = x

        return symbol

#encode
encoder = Encoder()
'''
probs = [0.1, 0.2, 0.3, 0.05, 0.05, 0.15, 0.05, 0.1]
data = arr
print('lenght', len(arr))

for symbol in data:
    encoder.encode_symbol(probs, symbol)
encoded = encoder.get_encoded()
print("The number of bytes needed is: " + str(len(encoded) * 4))

'''
num_bins = len(arr)
counts, bins = np.histogram(arr, bins=num_bins)
print('counts', counts)
probs = counts/float(counts.sum())
print('probs', probs)
probs = RAS.convert_float_array_to_list(probs)
probs = list(map(float, probs))
print('probs_1', probs)
print('array', arr)
print('type', type(arr))
#probs = [0.1, 0.2, 0.3, 0.05, 0.05, 0.15, 0.05, 0.1]

data = arr

for symbol in data:
    encoder.encode_symbol(probs, symbol)
encoded = encoder.get_encoded()
print("The number of bytes needed is: " + str(len(encoded) * 4))
length_decoded = len(data)

# Encoded is a list of 32-bit words.
# Do things afterwards with encoded - e.g. save it into a file or send it somewhere.

decoder = Decoder(encoded)
decoded_data = []
for _ in range(length_decoded):
    decoded_data.append(decoder.decode_symbol(probs))
print('decoded', decoded_data)

