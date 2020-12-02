"""Todo List
    - create class that writes converts array of ints to file: name it convert_int_array_to_file - done it's in RAS
    - create class that reads file and converts it to array: name it convert_file_to_int_array - done it's in RAS
    - create class that slices activations array to multiple arrays of given length (-what length to choose?) -
        work with
    - problem with the way activations are read from file where activations stored? need to create pathway or loop?
    - finally try look on NN:   -VGG19
                                -MobileNet
                                -ResNet"""





import struct
import numpy as np
import random
from math import floor, ceil




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


#Create Slice function



#function that slices float into x lenght int number - put in bracets size for size of int!
#put entries into an array
def slice_float(float):
    f_r = float
    f_s = str(float)
    d_l = f_s[::-1].find('.')
#    print('number of decimals', d_l)
    x = 1
    a = np.zeros(d_l + 1)
    while x < d_l+2:
        int_entry = int(f_r)
        a[x-1] = int_entry
        f_r = (10*f_r) - 10*floor(f_r)
        x += 1
    #issue with float since exact decimal isn't defined such that last number in array can come out wrong,
    #shouldn't be problem when using exact given number
    a = a.astype(int)
    return a

def max_float_lenght(float_array):
    n_max = len(str(float_array[0]))
    print('length of float array', n_max)
    x = 0
    while x < (len(float_array)-1):
        x += 1
        n_max = max(n_max, len(str(float_array[x])))
    length = n_max
    print('max_length', length, 'type', type(length))
    return length


def slice_float_array(f_array):
#first entry of each array is the length of the array to avoid taking with it 0's
    dim_1 = len(f_array)
    print('dim', dim_1)
    maxlength = max_float_lenght(f_array)
    print('maxlength', maxlength)
    s_f_a = np.zeros((dim_1, maxlength))
    length_array = np.zeros(dim_1)
    print('s_f_a', s_f_a)
    n = 0
    while n < (dim_1):
        f_a = slice_float(f_array[n])
#        print('f_a', f_a)
        m = 0
        length_array[n-1] = len(f_a)
        while m < (len(f_a)):
            s_f_a[n,m] = f_a[m]
            m += 1
        n += 1
#        print('n', n)
    print('s_f_a done', s_f_a, 'length s_f_a', len(s_f_a))
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