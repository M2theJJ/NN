# Julian Serra: jserra17@cmc.edu
# LZ77
#https://github.com/julyanserra/Basic-LZ77-in-Python/blob/master/encoder.py
'''
for encoder
x = name of file
y = size of max search buffer, i.e. 1024
'''

import struct
import sys
import math
import os
import numpy as np
from Extras import  RAS

arr = np.random.randint(1, 101, size=(100, 100))
#arr = RAS.convert_array_to_list(arr)
arr = RAS.convert_list_to_bytes(arr)



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

#try list
def e_main(list):
    # extra credit
    x = 16
    MAXSEARCH = 1024 #set to max search buffer idealy to 2^x
    MAXLH = int(math.pow(2, (x - (math.log(MAXSEARCH, 2)))))

#    file_to_read = sys.argv[1]
    input = arr
    searchiterator = 0;
    lhiterator = 0;

    while lhiterator < len(input):
        search = input[searchiterator:lhiterator]
        look_ahead = input[lhiterator:lhiterator + MAXLH]
        (offset, length, char) = LZ77_search(search, look_ahead)
        # print (offset, length, char)

        shifted_offset = offset << 6
        offset_and_length = shifted_offset + length
        ol_bytes = struct.pack(arr, offset_and_length, char)
        output.write(ol_bytes)

        lhiterator = lhiterator + length + 1
        searchiterator = lhiterator - MAXSEARCH

        if searchiterator < 0:
            searchiterator = 0

    file.close()


def parse(file):
    r = []
    f = open(file, "rb")
    text = f.read()
    return text


if __name__ == "__main__":
    e_main(arr)


'''-----------------------------------------------------------------------------------------------------------------------------------------------
for decoder
x = size of max search buffer used in encoder, i.e. 1024
y = file type, i.e. txt, jpg, etc.
'''


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


def d_main():
  MAX_SEARCH = int(sys.argv[1])
  file_type = sys.argv[2]
  processed = open("processed." + file_type, "wb")
  decoder("compressed.bin", processed, MAX_SEARCH)
  processed.close


if __name__ == "__main__":
  d_main()
