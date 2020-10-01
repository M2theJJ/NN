#encoder

import struct
import sys
import math
import numpy as np



arr = np.random.randint(1, 101, size=(3, 3))
print('array', arr)
bytes = bytes(arr)
print('bytes', bytes)
f = open("bytes.txt", "wb")
f.write(bytes)
f.close()
with open('bytes.txt', 'r') as f:
    print('read bytes', f.read())
text = np.savetxt("array.txt", arr, fmt="%s")
with open('array.txt', 'r') as f:
    print('read array', f.read())



def LZ77_search(search, look_ahead):
    ls = len(search)
    llh = len(look_ahead)
    print('search', search)
    print('look_ahead', look_ahead)
    print('ls', ls)
    print('llh', llh)
    if (ls == 0):
        return (0, 0, look_ahead[0])

    if (llh) == 0:
        return (-1, -1, "")

    best_length = 0
    best_offset = 0
    buf = search + look_ahead
    search_pointer = ls
    print( "search: " , search, " lookahead: ", look_ahead)
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
    print('we got here')

    return (best_offset, best_length, buf[search_pointer + best_length])


def main():
    # extra credit
    x = 16
    MAXSEARCH = int(sys.argv[2])
    MAXLH = int(math.pow(2, (x - (math.log(MAXSEARCH, 2)))))
    print('MAXSEARCH', MAXSEARCH)
    print('MAXLH', MAXLH)
    file_to_read = sys.argv[1]
    input = parse(file_to_read) # parse is function that opens file
    file = open("compressed.bin", "wb")
    searchiterator = 0;
    lhiterator = 0;

    while lhiterator < len(input):
        search = input[searchiterator:lhiterator]
        look_ahead = input[lhiterator:lhiterator + MAXLH]
        (offset, length, char) = LZ77_search(search, look_ahead)
        print ('offset, length, char',offset, length, char)

        shifted_offset = offset << 6
        offset_and_length = shifted_offset + length
        ol_bytes = struct.pack(">Hc", offset_and_length, char)
        file.write(ol_bytes)

        lhiterator = lhiterator + length + 1
        searchiterator = lhiterator - MAXSEARCH

        if searchiterator < 0:
            searchiterator = 0

    file.close()


def parse(file): #need to solve this - when putting in array.txt directly it workes but if sys.argv is array.txt that gets put in here it doesn't work
    r = []
    f = open('bytes.txt', "rb")
#    f = open('array.txt', "rb")
#    f = open(sys.argv[1], "rb")
    text = f.read()
    return text


if __name__ == "__main__":
    main()

#-----------------------------------------------------------------------------------------------------------------------

# decorder
import struct
import os
import sys


def decoder(name, out, search):
    MAX_SEARCH = search
    file = open(name, "rb")
    input = file.read()

    chararray = ""
    i = 0

    while i < len(input):

        # unpack, every 3 bytes (x,y,z)
        (offset_and_length, char) = struct.unpack(">Hc", input[i:i + 3])

        # shift right, get offset (length dissapears)
        offset = offset_and_length >> 6

        # substract by offset000000, gives length value
        length = offset_and_length - (offset << 6)

        # print "swag"
        # print offset
        # print length

        i = i + 3

        # case is (0,0,c)
        if (offset == 0) and (length == 0):
            chararray += char

        # case is (x,y,c)
        else:
            iterator = len(chararray) - MAX_SEARCH
            if iterator < 0:
                iterator = offset
            else:
                iterator += offset
            for pointer in range(length):
                chararray += chararray[iterator + pointer]
            chararray += char

    out.write(chararray)


def main():
    MAX_SEARCH = int(sys.argv[1]) #should be argv 2 from main encode
    file_type = sys.argv[2] #should be argv 1 from main encode
    processed = open("processed." + file_type, "wb")
    decoder("compressed.bin", processed, MAX_SEARCH)
    processed.close


if __name__ == "__main__":
    main()
