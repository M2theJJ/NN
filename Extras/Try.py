import struct
import numpy

f=open("myfile","wb")
mydata=numpy.random.random(10)
print(mydata)
myfmt='f'*len(mydata)
#  You can use 'd' for double and < or > to force endinness
bin=struct.pack(myfmt,*mydata)
print(bin)
f.write(bin)
f.close()