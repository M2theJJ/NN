from Extras import RAS
import numpy as np
import sys

def compress(uncompressed):
    """Compress a string to a list of output symbols."""

    # Build the dictionary.
    dict_size = 256
    #dictionary = dict((chr(i), chr(i)) for i in xrange(dict_size))
    dictionary = {chr(i): chr(i) for i in range(dict_size)}

    w = ""
    result = []
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            # Add wc to the dictionary.
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
            #print("dictionary", dictionary)

    # Output the code for w.
    if w:
        result.append(dictionary[w])
    return result

##check loop and dictionary
#16bit entry doesnt fit dictionary


def decompress(compressed):
    """Decompress a list of output ks to a string."""
    from io import StringIO

    # Build the dictionary.
    dict_size = 256
    #dictionary = dict((chr(i), chr(i)) for i in xrange(dict_size))
    dictionary = {chr(i): chr(i) for i in range(dict_size)}

    # use StringIO, otherwise this becomes O(N^2)
    # due to string concatenation in a loop
    result = StringIO()
    w = compressed.pop(0)
    result.write(w)
    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError('Bad compressed k: %s' % k)
        result.write(entry)

        # Add w+entry[0] to the dictionary.
        dictionary[dict_size] = w + entry[0]
        dict_size += 1

        w = entry
    return result.getvalue()


def LZ78_Do(array):

    print('LZ78:')
    list = RAS.convert_array_to_list(array)
    #print('list', list)
    string = RAS.convert_list_to_string(list)
    # How to use:
    #compressed = compress('TOBEORNOTTOBEORTOBEORNOT')
    compressed = compress(string)
    print('compressed', compressed)
#    compressed = RAS.convert_list_to_string(compressed)

#    s_c = RAS.get_size(compressed)
    print('type of:', type(compressed), 'length', len(compressed))
#    print ('compressed', compressed, 'lenght', len(compressed))
    decompressed = decompress(compressed)
#    s_d = RAS.get_size(decompressed)
    print ('decompressed', decompressed)
    print('tpe', type(decompressed), 'lenght', len(decompressed),)
    print('Ratio:', len(compressed) / len(decompressed))
#    print('Ratio of Bytes:', s_c/s_d)


#array = np.random.randint(1, 101, size=(100, 100))
#array = array.flatten()
#LZ = LZ78_Do(array)
#works properly with a ration < 1=