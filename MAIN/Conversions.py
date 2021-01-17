import binascii
import gc
import sys
import struct
import numpy as np
import random
from math import floor, ceil

# Creates random Array
'''
#array = np.random.randint(1, 101, 8)
#array = np.random.randint(101, 8) #array with 0s
array = np.random.randint(1, 101, size=(3, 3))
#array = [55, 33, 22]
#arr = 5 * np.random.random_sample((3, 2)) - 5 #from -5 to 0 / (b - a) * random_sample() + a
#array = [55, 33, 22]
#print (array)
f_array = np.random.uniform(low=0.5, high=13.3, size=(50,))
'''


# convert string to list
def convert_string_to_list(string):
    li = list(string.split(" "))
    return li


# Array to list
def convert_array_to_list(array):
    a = array
    a = a.flatten()
    #    print('flattened array', a)
    all_numbers = list()
    for number in a:
        number_str = "{0:d}".format(number)
        all_numbers.append(number_str)
    #    print('list', all_numbers)
    return all_numbers


# float Array to list
def convert_float_array_to_list(f_array):
    a = f_array
    a = a.flatten()
    print('flattened f_array', a)
    f_all_numbers = list()
    for number in a:
        number_str = "{0:f}".format(number)
        f_all_numbers.append(number_str)
    print('f_list', f_all_numbers)
    return f_all_numbers


# List to String
def convert_list_to_string(arr, seperator=' '):
    """ Convert list to string, by joining all item in list with given separator.
        Returns the concatenated string """
    # print('string', seperator.join(arr))
    return seperator.join(arr)


# Convert list of strings to string
# full_str = convert_list_to_string(arr)
# print('string', full_str)
# print('string', full_str)

# List to Byteslike
def convert_list_to_bytes(array):
    bytes_of_values = bytes(array)
    # print(bytes_of_values)b'7!\x16'
    # bytes_of_values == '\x37\x21\x16'
    # True
    print('bytes', bytes_of_values)
    return bytes_of_values


# bytes = convert_list_to_bytes(array)
# print('bytes', bytes)
# print('bytes', bytes)


# Hex to bytes
def convert_bytes_to_hex(bytes):
    hex = binascii.hexlify(bytes)
    print('hex', hex)
    return hex


# hex = convert_bytes_to_hex(bytes)
# print('hex', hex)


# Bytes to ASCII
def convert_bytes_to_ASCII(bytes):
    c = bytes
    #    c = b'\x0f\x00\x00\x00NR09G05164\x00'
    ASCII = c[0:100000000].decode("ascii")  # c[x:y], what is x and what is y?
    print('ASCII', ASCII)
    return ASCII


# ASCII = convert_bytes_to_ASCII(bytes)


# print('ASCII', ASCII)
# array to txt - filename in format: filename = filename.txt
def convert_array_to_txt(array, filename):
    text = np.savetxt(filename, array, fmt="%s")
    with open(filename, 'r') as f:
        print('read', f.read())
    return text


# text = convert_array_to_text(array)
# with open("array.txt", 'r') as f:
#    print('read', f.read())

# symbol count
def symb_count(array):
    list = convert_array_to_list(array)
    c_array = [list.count(x) for x in set(list)]
    print('c_array', c_array)
    return c_array


# convert Hex to list
def convert_hex_to_list(hex):
    list = list(binascii.unhexlify(hex))
    print('h_list', list(binascii.unhexlify(hex)))
    return list


# define filename like this first: trythis = "trythis.txt"
def convert_int_array_to_file(array, filename):
    # write w+ to create a new file each time!
    f = open(filename, "w")
    array = convert_list_to_string(convert_array_to_list(array))
    file = f.write(array)
    f.close()
    return file


def convert_file_to_int_array(fileName):
    fileObj = open(fileName, "r")  # opens the file in read mode
    array = fileObj.read().splitlines()  # puts the file into an array
    fileObj.close()
    with open(fileName, 'r') as file:
        string = file.read().replace('\n', '')
    li = list(string.split(" "))
    my_array = np.array(li)
    int_array = my_array.astype(int)
    return int_array


'''
trythis = "trythis.txt"
arr = array.flatten()
text = convert_int_array_to_file(arr, trythis)
arr = convert_file_to_int_array(trythis)
print('array from file', arr)
print('type of array from file', type(arr))
print('type of element in array', type(arr[0]))


#somehow read float from text file to something like this
def convert_file_to_float_array(fileName):
    data = np.genfromtxt(fileName, usecols=1, dtype=float)
    return data
'''


def convert_file_to_float_array(fileName):
    fileObj = open(fileName, "r")  # opens the file in read mode
    array = fileObj.read().splitlines()  # puts the file into an array
    fileObj.close()
    with open(fileName, 'r') as file:
        string = file.read().replace('\n', '')
    li = list(string.split(" "))
    my_array = np.array(li)
    int_array = my_array.astype(float)
    return int_array


def convert_f_array_to_file(array, filename):
    # write w+ to create a new file each time!
    f = open(filename, "w")
    array = convert_list_to_string(convert_float_array_to_list(array))
    file = f.write(array)
    f.close()
    return file


'''
trythis = "trythis.txt"

text = convert_f_array_to_file(f_array, trythis)
d_f = convert_file_to_float_array(trythis)
print('floatarray', d_f, 'type', type(d_f))
'''


def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz


# from here on it's try code

"""
#try to write array to file gives out: error!
arr = np.random.randint(1, 101, size=(3, 3))
arr = arr.flatten()
print('type of array @ beginning', type(arr))
#try write list to file: error!
#------> need to convert array to string!
#with given array this writes array to file in str format
#input the string works as planned
trythis = "trythis.txt"
def convert_int_array_to_file(array, filename):
    #write w+ to create a new file each time!
    f=open(filename,"w")
    array = RAS.convert_list_to_string(RAS.convert_array_to_list(array))
    file = f.write(array)
    f.close()
    return file
text = convert_int_array_to_file(arr, trythis)
#works to this point!

#try to read file and convert it into array again.
#returns list for now - need to get back an array to work with
def convert_file_to_int_array(fileName):
    fileObj = open(fileName, "r")  # opens the file in read mode
    array = fileObj.read().splitlines()  # puts the file into an array
    fileObj.close()
    with open(fileName, 'r') as file:
        string = file.read().replace('\n', '')
    li = list(string.split(" "))
    my_array = np.array(li)
    int_array = my_array.astype(int)
    return int_array


arr = convert_file_to_int_array(trythis)
print('array from file', arr)
print('type of array from file', type(arr))
print('type of element in array', type(arr[0]))

#-------------------------------------------------

def txt_2_string(filename):
    with open(filename, 'r') as file:
        data = file.read().replace('\n', '')
    return data

#data = txt_2_string(trythis)
#print('data', data, 'type of data', type(data))

def string_2_list(string):
    li = list(string.split(" "))
    return li
# Driver code
#li = string_2_list(data)
#print('list', string_2_list(data))



#turns list into np.array but with single entry:
#my_array = np.array(array)
#my_array = np.array(li)
#print('myarray', my_array)
#print('type', type(my_array))
#print('type of entry', type(my_array[0]))
#int_array = my_array.astype(int)
#print('int_array', int_array)
#print('int_array type element', type(int_array[0]))


"""


# Create Slice function


# function that slices float into x lenght int number - put in bracets size for size of int!
# put entries into an array
def slice_float(float):
    f_r = float
    f_s = str(float)
    d_l = f_s[::-1].find('.')
    #    print('number of decimals', d_l)
    x = 1
    a = np.zeros(d_l + 1)
    while x < d_l + 2:
        int_entry = int(f_r)
        a[x - 1] = int_entry
        f_r = (10 * f_r) - 10 * floor(f_r)
        x += 1
    # issue with float since exact decimal isn't defined such that last number in array can come out wrong,
    # shouldn't be problem when using exact given number
    a = a.astype(int)
    return a


def max_float_lenght(float_array):
    n_max = len(str(float_array[0]))
    #    print('float_array[0]', float_array[0],'bin', binary(float_array[0]))
    #    print('length of float array', n_max)
    x = 0
    while x < (len(float_array) - 1):
        x += 1
        n_max = max(n_max, len(str(float_array[x])))
    length = n_max
    #    print('max_length', length, 'type', type(length))
    return length


def slice_float_array(f_array):
    # first entry of each array is the length of the array to avoid taking with it 0's
    dim_1 = len(f_array)
    #    print('dim', dim_1)
    maxlength = max_float_lenght(f_array)
    #    print('maxlength', maxlength)
    s_f_a = np.zeros((dim_1, maxlength))
    length_array = np.zeros(dim_1)
    #    print('s_f_a', s_f_a)
    n = 0
    while n < (dim_1):
        f_a = slice_float(f_array[n])
        #        print('f_a', f_a, 'type', type(f_a))
        m = 0
        length_array[n - 1] = len(f_a)
        while m < (len(f_a)):
            s_f_a[n, m] = f_a[m]
            m += 1
        n += 1
    #        print('n', n)
    #    print('s_f_a done', s_f_a, 'length s_f_a', len(s_f_a))
    return s_f_a, length_array


'''
f = random.uniform(1.5, 9.5)
print('float:', f)
f_array = np.random.uniform(low=0.5, high=9.3, size=(50, 3))
f_array = f_array.flatten()
print('float array', f_array)
print('length of array', len(f_array))
#max_length = max_float_lenght(f_array)
s_float_array = slice_float_array(f_array)
print('check s_f_a:', s_float_array[0], 'check length array:', s_float_array[1])
'''


def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))


# biString = binary(0.32)
# print(biString, type(biString), int(biString))

def bin_array(array):
    array = array.flatten()
    b_array = np.zeros((len(array)))
    for x in range(len(array)):
        f = array[x]
        b = binary(f)
        b_array[x] = b
        x += 1
    return b_array


# f_array = np.random.uniform(low=0.5, high=9.3, size=(50, 50))
# b = bin_array(f_array)
# print(b, b.dtype)


def sliceF(float):
    biString = binary(float)
    x_0 = 0
    x_1 = 8
    arraysliced = np.array([0, 0, 0, 0], dtype=np.int8)
    for x in range(int(len(biString) / 8)):
        slice_x = biString[x_0:x_1]
        #        print('slice from', x_0, 'to', x_1, slice_x)
        x_0 = x_1
        x_1 += 8
        s_s_x = str(slice_x)
        entry = int(s_s_x, 2)
        arraysliced[x] = entry
        # arraysliced[x] = int(slice_x)
    # print(arraysliced)
    return arraysliced


# input float array
def sliceFA(f_array):
    f_array = f_array.flatten()
    dim_1 = len(f_array)
    #    print('dim', dim_1)
    maxlength = 4
    #    print('maxlength', maxlength)
    s_f_a = np.zeros((dim_1, maxlength), dtype=np.int8)
    length_array = np.zeros(dim_1)
    #    print('s_f_a', s_f_a)
    n = 0
    while n < (dim_1):
        f_a = sliceF(f_array[n])
        #        print('f_a', f_a, 'type', type(f_a))
        m = 0
        length_array[n - 1] = len(f_a)
        while m < (len(f_a)):
            s_f_a[n, m] = f_a[m]
            m += 1
        n += 1
    #        print('n', n)
    #    print('s_f_a done', s_f_a, 'length s_f_a', len(s_f_a))
    return s_f_a, length_array


# f = random.uniform(1.5, 9.5)
# f_array = np.random.uniform(low=0.5, high=9.3, size=(50, 50))
# s_a = sliceFA(f_array)
# print(s_a[0].dtype)

# f = random.uniform(1.5, 9.5)
# print('f', f)
# slice = sliceF(f)
# print('slice', slice)
# f_array = np.random.uniform(low=0.5, high=9.3, size=(50, 3))
# f_array = f_array.flatten()
# print(sys.getsizeof())
# sliceA = sliceFA(f_array)[0].flatten()
# print('result', sliceA, 'length', len(sliceA))

# check LZ77 again to see if one can use that procedure
def size_of_data(input, type):
    # transform input to np.array
    input = np.array(input).flatten

    # count number of entries
    no_e = len(input)

    # get datatype & it's size
    type_e = input.dtype
    if type_e == 'int8':
        s_e = 8
    if type_e == 'int16':
        s_e = 16
    if type_e == 'int32':
        s_e = 32
    if type_e == 'float32':
        s_e = 32

    # get size of data
    size = no_e * s_e
    print()
    return size


def save_array(filename, array):
    saved = np.save(filename, array)
    return saved


# filename.npy maybe needed to add
def load_array(filename):
    loaded = np.load(filename)
    return loaded


'''
x = np.random.randint(1, 101, size=(3, 3))
saved = save_array("saved_try", x)
print('array before', x, 'array loaded', np.load("saved_try.npy"), 'saved', saved)
'''
