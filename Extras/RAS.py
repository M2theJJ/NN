import numpy as np
import binascii

#Creates random Array

#array = np.random.randint(1, 101, 8)
#array = np.random.randint(101, 8) #array with 0s
array = np.random.randint(1, 101, size=(3, 3))
#array = [55, 33, 22]
#arr = 5 * np.random.random_sample((3, 2)) - 5 #from -5 to 0 / (b - a) * random_sample() + a
#array = [55, 33, 22]
#print (array)
f_array = np.random.uniform(low=0.5, high=13.3, size=(50,))


#Array to list
def convert_array_to_list(array):
    a = array
    a = a.flatten()
    print('flattened array', a)
    all_numbers = list()
    for number in a:
        number_str = "{0:d}".format(number)
        all_numbers.append(number_str)
    print('list', all_numbers)
    return all_numbers


#float Array to list
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
    



#List to String
def convert_list_to_string(arr, seperator=' '):
    """ Convert list to string, by joining all item in list with given separator.
        Returns the concatenated string """
    print('string', seperator.join(arr))
    return seperator.join(arr)
# Convert list of strings to string
#full_str = convert_list_to_string(arr)
#print('string', full_str)
#print('string', full_str)

#List to Byteslike
def convert_list_to_bytes(array):

    bytes_of_values = bytes(array)
    #print(bytes_of_values)b'7!\x16'
    #bytes_of_values == '\x37\x21\x16'
    #True
    print('bytes', bytes_of_values)
    return bytes_of_values
#bytes = convert_list_to_bytes(array)
#print('bytes', bytes)
#print('bytes', bytes)



#Hex to bytes
def convert_bytes_to_hex(bytes):
    hex = binascii.hexlify(bytes)
    print('hex', hex)
    return hex
#hex = convert_bytes_to_hex(bytes)
#print('hex', hex)


#Bytes to ASCII
def convert_bytes_to_ASCII(bytes):
    c = bytes
#    c = b'\x0f\x00\x00\x00NR09G05164\x00'
    ASCII = c[0:100000000].decode("ascii") #c[x:y], what is x and what is y?
    print('ASCII', ASCII)
    return ASCII
#ASCII = convert_bytes_to_ASCII(bytes)


#print('ASCII', ASCII)
#array to txt - filename in format: filename = filename.txt
def convert_array_to_txt(array, filename):
    text = np.savetxt(filename, array, fmt="%s")
    with open(filename, 'r') as f:
        print('read', f.read())
    return text
#text = convert_array_to_text(array)
#with open("array.txt", 'r') as f:
#    print('read', f.read())

#symbol count
def symb_count(array):
    list = convert_array_to_list(array)
    c_array = [list.count(x) for x in set(list)]
    print('c_array', c_array)
    return c_array

#convert Hex to list
def convert_hex_to_list(hex):
    list = list(binascii.unhexlify(hex))
    print('h_list', list(binascii.unhexlify(hex)))
    return list

#define filename like this first: trythis = "trythis.txt"
def convert_int_array_to_file(array, filename):
    #write w+ to create a new file each time!
    f=open(filename,"w")
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


trythis = "trythis.txt"
arr = array.flatten()
text = convert_int_array_to_file(arr, trythis)
arr = convert_file_to_int_array(trythis)
print('array from file', arr)
print('type of array from file', type(arr))
print('type of element in array', type(arr[0]))

'''
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
    #write w+ to create a new file each time!
    f=open(filename,"w")
    array = convert_list_to_string(convert_float_array_to_list(array))
    file = f.write(array)
    f.close()
    return file

trythis = "trythis.txt"

text = convert_f_array_to_file(f_array, trythis)
d_f = convert_file_to_float_array(trythis)
print('floatarray', d_f, 'type', type(d_f))