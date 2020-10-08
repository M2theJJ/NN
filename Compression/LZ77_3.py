#https://github.com/ThomasWWebb/LZ77-Compression/blob/master/LZ77.py

import sys
from Extras import bitstring
from Extras import RAS
import numpy as np

arr = np.random.randint(1, 101, size=(100, 100))
arr = arr.flatten()
array = RAS.convert_array_to_list(arr)
#text = RAS.convert_list_to_string(arr)


def lz77(textFile, windowBits, lengthBits):
    data = bitstring.BitArray(textFile)
    print('data', data)
    print('data', data.int)
    compressed = compress(data, windowBits, lengthBits)
    decompressed = decompress(compressed, windowBits, lengthBits)

    print("Data length: {} bits".format(len(data)))
    print("Compressed length: {} bits".format(len(compressed)))

    if (data == decompressed):
        print("Successful compression and decompression")
    else:
        print("Error in compression and decompression")


def compress(data, windowBits, lengthBits):
    maxWindowLength = 2 ** windowBits - 1
    bufferLength = 2 ** lengthBits - 1
    buffer = data[:bufferLength]
    substring = buffer
    compressed = bitstring.BitArray()
    window = bitstring.BitArray('')

    # constants in the case that a match is not found
    zeroPos = bitstring.Bits(uint=0, length=windowBits)
    zeroLength = bitstring.Bits(uint=0, length=lengthBits)

    bufferPos = 0
    maxLength = len(data)
    while ((bufferPos) < maxLength):
        bufferExtend = min(bufferPos + bufferLength, maxLength)
        buffer = data[bufferPos: bufferExtend]
        bufferStepper = len(buffer)
        tripletAdded = False
        while bufferStepper > 0 and not tripletAdded:
            substring = buffer[0:bufferStepper]

            if (window.find(substring) != ()):
                position = len(window) - int(window.find(substring)[0])

                length = len(substring)
                nextCharIndex = bufferPos + length
                if nextCharIndex > len(data) - 1:
                    nextCharIndex -= 1
                    substring = substring[:-1]
                    length = len(substring)
                nextChar = data[nextCharIndex:nextCharIndex + 1]

                bitsPosition = bitstring.Bits(uint=position, length=windowBits)
                bitsLength = bitstring.Bits(uint=length, length=lengthBits)

                compressedTriplet = bitsPosition + bitsLength + nextChar
                substring += nextChar
                length += 1
                tripletAdded = True

            elif len(substring) == 1:
                length = 1
                compressedTriplet = zeroPos + zeroLength + substring
                print('comrpessedTriplet', compressedTriplet)
                tripletAdded = True
            bufferStepper -= 1
        bufferPos += length
        window += substring

        if len(window) > maxWindowLength:
            startIndex = len(window) - maxWindowLength
            window = window[startIndex:]
        compressed += compressedTriplet

    print('compressed?:', compressed)
    print('type', type(compressed))
    print('int_c', compressed.int)
    return compressed


def decompress(compressed, windowBits, lengthBits):
    decompressedData = bitstring.BitArray('')
    i = 0
    while i in range(len(compressed)):
        pos = compressed[i:(i + windowBits)].uint
        i += windowBits
        length = compressed[i:(i + lengthBits)].uint
        i += lengthBits
        char = compressed[i:(i + 1)]
        i += 1
        if (pos == 0 and length == 0):
            decompressedData += char
        else:
            startPos = len(decompressedData) - pos
            endPos = startPos + length
            substring = decompressedData[startPos:endPos] + char
            decompressedData += substring

    print('decompressed?', decompressedData)
    print('type', type(decompressedData))
    h_l = decompressedData.bytes
    print('h_l', h_l)

    print('Ration:', len(compressed) / len(decompressedData))
    return decompressedData




if __name__ == "__main__":
    lz77(arr, 8, 8)
