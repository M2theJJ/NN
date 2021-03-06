#https://github.com/DyakoVlad/python-LZ78
import numpy as np

array = np.random.randint(1, 101, size=(3, 3))
#array = 5 * np.random.random_sample((3, 2)) - 5
#array = array.flatten()
print(array)
file = text = np.savetxt("stringIn.txt", array, fmt="%s")
EncodeIn = 'stringIn.txt'
EndcodeOut = 'stringOut.txt'
DecodeOut = 'd_stringOut.txt'

def encodeLZ(stringIn, stringOut):
    input_file = open(stringIn, 'r')
    encoded_file = open(stringOut, 'w')
    text_from_file = input_file.read()
    dict_of_codes = {text_from_file[0]: '1'}
    encoded_file.write('0' + text_from_file[0])
    text_from_file = text_from_file[1:]
    combination = ''
    code = 2
    for char in text_from_file:
        combination += char
        if combination not in dict_of_codes:
            dict_of_codes[combination] = str(code)
            if len(combination) == 1:
                encoded_file.write('0' + combination)
            else:
                encoded_file.write(dict_of_codes[combination[0:-1]] + combination[-1])
            code += 1
            combination = ''
    print('dict of code', dict_of_codes)
    input_file.close()
    encoded_file.close()
    return dict_of_codes

##use dictionary used in encode pass to decode store dictionary in encode and put it in decode


def decodeLZ(FileIn, FileOut, dict_of_codes):
    coded_file = open(FileIn, 'r')
    decoded_file = open(FileOut, 'w')
    text_from_file = coded_file.read()
    #my try
    dict_of_codes = dict_of_codes
#original    dict_of_codes = {'0': '', '1': text_from_file[1]}
    decoded_file.write(dict_of_codes['1'])
    text_from_file = text_from_file[2:]
    combination = ''
    code = 2
    for char in text_from_file:
        if char in '1234567890' or ' ':
            print('if', char)
            combination += char
        else:
            print('else', char)
            dict_of_codes[str(code)] = dict_of_codes[combination] + char
            decoded_file.write(dict_of_codes[combination] + char)
            combination = ''
            code += 1
    coded_file.close()
    decoded_file.close()


#encodeLZ('input.txt', 'encoded.txt')
#decodeLZ('encoded.txt', 'decoded.txt')

#encodeLZ(EncodeIn, EndcodeOut)
dict_of_codes = encodeLZ(EncodeIn, EndcodeOut)
decodeLZ(EndcodeOut, DecodeOut, dict_of_codes)
