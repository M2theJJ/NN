import numpy as np
from Compression import Huffman01

def random_array_to_string():
    array = np.random.randint(1, 101, 8)

    print (array)

# ---------------------------------------

#flatten array - put in array name
#'array'.flatten()

#---------------------------------------

#convert array to ascii
    print ("".join([chr(item) for item in array]))

# ---------------------------------------

#convert array to list
#    array = list(map(str,array))

#    print (array)
# ---------------------------------------


#    output = "";

#    for item in array:
#        output += ''
#        output += item
#        output += ' '

#    print(output)

# ---------------------------------------

#    separator = ' '
#    print(separator.join(array))

#---------------------------------------

#convert array to string
#    str1 = ''.join(str(e) for e in array)

#    print (str1)


# ---------------------------------------

# convert array to list
#    str1  = list(map(str, str1))

#    print(str1)

#---------------------------------------

random_array_to_string()
#h = Huffman01.HuffmanCoding(array)

#output_path = h.compress()
#h.decompress(output_path)

#from huffman import HuffmanCoding

#input file path
#path = "/home/ubuntu/Downloads/sample.txt"

#h = HuffmanCoding(path)

#output_path = h.compress()
#h.decompress(output_path)