import numpy as np
import binascii

#Creates random Array

#array = np.random.randint(1, 101, 8)
#array = np.random.randint(101, 8) #array with 0s
array = np.random.randint(1, 101, size=(3, 3))
#array = [55, 33, 22]
#arr = 5 * np.random.random_sample((3, 2)) - 5 #from -5 to 0 / (b - a) * random_sample() + a
array = np.random.randint(1, 101, size=(3, 3))
#array = [55, 33, 22]
#print (array)


#Array to String
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

arr = convert_array_to_list(array)


#List to String
def convert_list_to_string(arr, seperator=' '):
    """ Convert list to string, by joining all item in list with given separator.
        Returns the concatenated string """
    return seperator.join(arr)
# Convert list of strings to string
full_str = convert_list_to_string(arr)
#print('string', full_str)
print('string', full_str)

#List to Byteslike
def convert_list_to_bytes(array):

    bytes_of_values = bytes(array)
    #print(bytes_of_values)b'7!\x16'
    #bytes_of_values == '\x37\x21\x16'
    #True
    return bytes_of_values
bytes = convert_list_to_bytes(array)
#print('bytes', bytes)
print('bytes', bytes)



#Hex to bytes
def convert_bytes_to_hex(bytes):
    hex = binascii.hexlify(bytes)
    return hex
hex = convert_bytes_to_hex(bytes)
#print('hex', hex)
print('hex', hex)

#Bytes to ASCII
def convert_bytes_to_ASCII(bytes):
    c = bytes
#    c = b'\x0f\x00\x00\x00NR09G05164\x00'
    ASCII = c[0:100000000].decode("ascii") #c[x:y], what is x and what is y?
    return ASCII
ASCII = convert_bytes_to_ASCII(bytes)
#print('ASCII', ASCII)
print('ASCII', ASCII)

#array to txt
def convert_array_to_txt(array):
    text = np.savetxt("array.txt", array, fmt="%s")
    return text
text = convert_array_to_txt(arr)
print(type(text))
print('text', "array.txt")

#txt to array
def convert_txt_to_array(fileName): #fileName in array to txt definded
        fileObj = open(fileName, "r")  # opens the file in read mode
        words = fileObj.read().splitlines()  # puts the file into an array
        fileObj.close()
        return words
a_f_t = convert_txt_to_array("array.txt")
print('array from text', a_f_t)