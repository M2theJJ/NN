#https://github.com/bits-back/bits-back/blob/master/rans.py
# Feed an Array to flatten the feed msg to encoder
"""
Closely based on https://github.com/rygorous/ryg_rans/blob/master/rans64.h by
Fabian Giesen.
We use the pythonic names `append` and `pop` for encoding and decoding
respectively. The compressed state is a pair `msg = (head, tail)`, where `head`
is an int in the range `[0, 2 ** head_precision)` and `tail` is an immutable
stack, implemented using a cons list, containing ints in the range
`[0, 2 ** tail_precision)`. The precisions must satisfy
  tail_precision < head_precision <= 2 * tail_precision.
For convenient compatibility with Numpy dtypes we use the settings
head_precision = 64 and tail_precision = 32.
Both the `append` method and the `pop` method assume access to a probability
distribution over symbols. We use the name `symb` for a symbol. To describe the
probability distribution we model the real interval [0, 1] with the range of
integers {0, 1, 2, ..., 2 ** precision}. Each symbol is represented by a
sub-interval within that range. This can be visualized for a probability
distribution over the set of symbols {a, b, c, d}:
    0                                                             1
    |          |----- P(symb) ------|                             |
    |                                                             |
    |    a           symb == b           c              d         |
    |----------|--------------------|---------|-------------------|
    |                                                             |
    |          |------ prob --------|                             |
    0        start                                            2 ** precision
Each sub-interval can be represented by a pair of non-negative integers:
`start` and `prob`. As shown in the above diagram, the number `prob` represents
the width of the interval, corresponding to `symb`, so that
  P(symb) = prob / 2 ** precision
where P is the probability mass function of our distribution.
The number `start` represents the beginning of the interval corresponding to
`symb`, which is analagous to the cumulative distribution function evaluated on
`symb`.
"""
import numpy as np
from functools import reduce
from Extras import RAS


head_precision = 64
tail_precision = 32
tail_mask = (1 << tail_precision) - 1
head_min  = 1 << head_precision - tail_precision

#          head    , tail
msg_init = head_min, ()

def append(msg, start, prob, precision): #encoding
    """
    Encodes a symbol with range `[start, start + prob)`.  All `prob`s are
    assumed to sum to `2 ** precision`. Compressed bits get written to `msg`.
    """
    # Prevent Numpy scalars leaking in
    start, prob, precision = map(int, [start, prob, precision])
    head, tail = msg
    if head >= prob << head_precision - precision:
        # Need to push data down into tail
        head, tail = head >> tail_precision, (head & tail_mask, tail)
    return (head // prob << precision) + head % prob + start, tail

def pop(msg, statfun, precision): #decoding
    """
    Pops a symbol from msg. The signiature of statfun should be
        statfun: cf |-> symb, (start, prob)
    where `cf` is in the interval `[start, start + prob)` and `symb` is the
    symbol corresponding to that interval.
    """
    # Prevent Numpy scalars leaking in
    precision = int(precision)
    head, tail = msg
    cf = head & ((1 << precision) - 1)
    symb, (start, prob) = statfun(cf)
    # Prevent Numpy scalars leaking in
    start, prob = int(start), int(prob)
    head = prob * (head >> precision) + cf - start
    if head < head_min:
        # Need to pull data up from tail
        head_new, tail = tail
        head = (head << tail_precision) + head_new
    return (head, tail), symb

def append_symbol(statfun, precision):
    def append_(msg, symbol):
        start, prob = statfun(symbol)
        return append(msg, start, prob, precision)
    return append_

def pop_symbol(statfun, precision):
    def pop_(msg):
        return pop(msg, statfun, precision)
    return pop_

def flatten(msg):
    """Flatten a rANS message into a 1d numpy array."""
    out, msg = [msg[0] >> 32, msg[0]], msg[1]
    while msg:
        x_head, msg = msg
        out.append(x_head)
    return np.asarray(out, dtype=np.uint32)

def unflatten(arr):
    """Unflatten a 1d numpy array into a rANS message."""
    return (int(arr[0]) << 32 | int(arr[1]),
            reduce(lambda tl, hd: (int(hd), tl), reversed(arr[2:]), ()))
'''
def test_rans():
    x = msg_init
    print('x_0', x)
    scale_bits = 8
    starts = rng.randint(0, 256, size=1000)
    freqs = rng.randint(1, 256, size=1000) % (256 - starts)
    freqs[freqs == 0] = 1
    assert np.all(starts + freqs <= 256)
    print("Exact entropy: " + str(np.sum(np.log2(256 / freqs))) + " bits.")
    # Encode
    for start, freq in zip(starts, freqs):
        x = append(x, start, freq, scale_bits)
    coded_arr = flatten(x)
    print('x_1', x)
    assert coded_arr.dtype == np.uint32
    print("Actual output size: " + str(32 * len(coded_arr)) + " bits.")

    # Decode
    x = unflatten(coded_arr)
    for start, freq in reversed(list(zip(starts, freqs))):
        def statfun(cf):
            assert start <= cf < start + freq
            return None, (start, freq)
        x, symbol = pop(x, statfun, scale_bits)
    print('head_min', head_min)
    print('x_2', x)
    assert x == (head_min, ())



def test_flatten_unflatten():
    state = msg_init
    some_bits = rng.randint(1 << 8, size=5)
    for b in some_bits:
        state = append(state, b, 1, 8)
    flat = flatten(state)
    state_ = unflatten(flat)
    flat_ = flatten(state_)
    assert np.all(flat == flat_)

#Test
rng = np.random.RandomState(0)

#test = test_rans()

#test_f_u = test_flatten_unflatten()
'''
rng = np.random.RandomState(0)
def rANS_Do(array):

    x = array
    original =array
#    print('x_arr', x)

    input = 32 * len(x)
#    print('input', input)
    x = unflatten(x)
#    print("Actual output size of x: " + str(32 * len(x)) + " bits.")
#    print('x_0', x)
    scale_bits = 8
    starts = rng.randint(0, 256, size=1000)
#    print('starts', starts, 'starts length', len(starts))
    freqs = rng.randint(1, 256, size=1000) % (256 - starts)
#    print('freqs', freqs, 'length of freq', len(freqs))
    freqs[freqs == 0] = 1
    assert np.all(starts + freqs <= 256)
#    print("Exact entropy: " + str(np.sum(np.log2(256 / freqs))) + " bits.")
    # Encode
    for start, freq in zip(starts, freqs):
        x = append(x, start, freq, scale_bits)
    coded_arr = flatten(x)
    #coded_arr_1 = unflatten(coded_arr)
#    print('coded_array_1', coded_arr_1)
#    print('x_1', x)
    assert coded_arr.dtype == np.uint32
#    print("Actual output size of encoded: " + str(32 * len(coded_arr)) + " bits.")
    encoded = 32 * len(coded_arr)
#    print('encoded', encoded)

#    print('Ratio:', len(encoded) / len(input))

    # Decode
    x = unflatten(coded_arr)
    for start, freq in reversed(list(zip(starts, freqs))):
        def statfun(cf):
            assert start <= cf < start + freq
            return None, (start, freq)
        x, symbol = pop(x, statfun, scale_bits)
#    print('head_min', head_min)
#    print('x_2', x)
    f = flatten(x)
#    print('decoded', f)
    assert original.all() == f.all()

#test = rANS_Do(np.random.randint(0,9,512))
#test = rANS_Do(np.random.randint(1,9, (10,10)).flatten())
#test_1 = test_rans_1()
'''
array = np.random.randint(1, 128, size=(3, 3))
print('array', array)
array = array.flatten()
print('1D array', array)
msg = unflatten(array)
print('array into rANS msg', msg)
array = flatten(msg)
print('rANS msg into array', array)
precision = 7 #set to 2**7 = 128 = range of rand.int.array - not quite sure how precision settings change compression rate
e = append(msg, 0, 8, precision)
print('encoded', e)
statfun = 1.0
d = pop(msg, statfun, precision)

#list = RAS.convert_array_to_list(array)

#check the test_rans to see where problem lies exactly and fix
'''