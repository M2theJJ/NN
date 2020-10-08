# Julian Serra: jserra17@cmc.edu
# LZ77

import struct
import sys
import math
import numpy as np
from Extras import RAS


def LZ77_search(search, look_ahead):
    ls = len(search)
    llh = len(look_ahead)

    if (ls == 0):
        return (0, 0, look_ahead[0])

    if (llh) == 0:
        return (-1, -1, "")

    best_length = 0
    best_offset = 0
    buf = search + look_ahead

    search_pointer = ls
    # print( "search: " , search, " lookahead: ", look_ahead)
    for i in range(0, ls):
        length = 0
        while buf[i + length] == buf[search_pointer + length]:
            length = length + 1
            if search_pointer + length == len(buf):
                length = length - 1
                break
            if i + length >= search_pointer:
                break
        if length > best_length:
            best_offset = i
            best_length = length

    return (best_offset, best_length, buf[search_pointer + best_length])


def main(input, m_s):
    #input = file, m_s = maxsearchbuffer
    # extra credit
    x = 16
    MAXSEARCH = int(m_s)
    MAXLH = int(math.pow(2, (x - (math.log(MAXSEARCH, 2)))))

    list = RAS.convert_array_to_list(input)
    file = RAS.convert_list_to_string(list)
#    file = RAS.convert_list_to_bytes(array)
#    file = RAS.convert_bytes_to_ASCII(file)
    searchiterator = 0;
    lhiterator = 0;

    while lhiterator < len(input):
        search = input[searchiterator:lhiterator]
        look_ahead = input[lhiterator:lhiterator + MAXLH]
        (offset, length, char) = LZ77_search(search, look_ahead)
        print ('look here',offset, length, char)

        shifted_offset = offset << 6
        offset_and_length = shifted_offset + length
        print('type char', type(char))
#        char = char.encode('ASCII')
        ol_bytes = struct.pack(">Hc", offset_and_length, char)
#        file.write(ol_bytes)

        lhiterator = lhiterator + length + 1
        searchiterator = lhiterator - MAXSEARCH

        if searchiterator < 0:
            searchiterator = 0

    file.close()


#def parse(file):
    #convert array to list? or to string?
    '''
    r = []
    f = open(file, "rb")
    text = f.read()
    return text'''
array = np.random.randint(1, 101, size=(3, 3))
array = array.flatten()
list = RAS.convert_array_to_list(array)
string = RAS.convert_list_to_string(list)

if __name__ == "__main__":
    main(array, 8)
