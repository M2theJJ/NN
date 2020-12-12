import logging
from typing import List, Tuple
import numpy as np
from Extras import RAS
from Extras import Try
import struct
from Compression_TO import Huffman, LZ77, LZ78, RLE, rANS, bzip2
import os, sys
import os.path
from os import path

#path aviable by copying from prject and putting in this:
#path = '/home/manjos/PycharmProjects/NN/NN_TO/activations_VGG19.txt'
path = '/Users/m2thejj/PycharmProjects/NN/NN_TO/activations_VGG19.txt'
with open(path) as f:
    mylist = list(f)
print('my list', mylist)


sliced_array = Try.sliceFA(RAS.convert_file_to_float_array(path))
a = sliced_array[0]
print('this is a', a)
int_s_a = a.astype(int)
print('this is int_s_a', a.astype(int))
#we have int array consisting of arrays now
int_s_a = int_s_a.flatten()
print('array', int_s_a)
print('lenght', len(int_s_a))

#huff = Huffman.Huffman_Do(int_s_a)
#lz77 = LZ77.LZ77_Do(int_s_a)
#lz78 = LZ78.LZ78_Do(int_s_a)
#rans = rANS.rANS_Do(int_s_a)
#rle = RLE.RLE_Do(int_s_a)
#bzip_2 = bzip2.bzip2_Do(int_s_a)






'''

def main():

   print ("File exists:"+str(path.exists('sliced_activations_VGG19.txt')))
   print ("File exists:" + str(path.exists('career.guru99.txt')))
   print ("directory exists:" + str(path.exists('NN_TO')))

if __name__== "__main__":
   main()



does_it = path.exists("sliced_activations_VGG19.txt")
print(does_it)



script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "sliced_activations_VGG19.txt"
abs_file_path = os.path.join(script_dir, rel_path)
print(abs_file_path)

fp = open(abs_file_path, 'r+');

info=[]
with open(os.('/home/manjos/PycharmProjects/NN/Compression_TO/sliced_activations_VGG19.txt')) as myfile:
    for line in myfile:
       info.append(line)

'''
