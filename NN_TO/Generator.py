import numpy as np
import sys
import os
import tensorflow as tf
import keras
from keras.models import load_model
'''
#when loading global weights add allow_pickle = True!
a99 = np.load('ResNetv2global_weights.npy', allow_pickle=True)
print('output:', a99[0].shape)
print('length:', len(a99))
print('length 0:', len(a99[0]))
print('length 0,0:', len(a99[0][0]))
print('length 0,0,0:', len(a99[0][0][0]))
print('length 0,0,0,0:', len(a99[0][0][0][0]))
#print('length 0,0,0,0,0:', len(a99[0][0][0][0][0])) ab hier float'''

#\Users\josep\PycharmProjects\NN\NN_TO\saved_models\cifar10_ResNet29v2_model.{epoch:03d}.h5

loaded_model = load_model("resnetv2_model.h5")
print(loaded_model)