import numpy as np

#array = np.array([[1,2], [3,4]])
array = np.random.randint(1, 101, size=(3, 3))
#array = np.random.randint(1, 101, 8) #for 1 dimension

#1. flatten arr. to vec.
#2. each entry into char.

def W_C_tran(array):

    a = array
    print(a)
    a = a.flatten()
    np.ravel(a)
    print('a', a)
    a = list(map(str,a))
    print(a)
    return a

arr = W_C_tran(array)

#print(type(arr))



import binascii

def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int(binascii.hexlify(text.encode(encoding, errors)), 16))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return int2bytes(n).decode(encoding, errors)

def int2bytes(i):
    hex_string = '%x' % i
    n = len(hex_string)
    return binascii.unhexlify(hex_string.zfill(n + (n & 1)))


c = b'\x0f\x00\x00\x00NR09G05164\x00'
c[4:8].decode("ascii")

print(c)


