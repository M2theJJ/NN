from __future__ import print_function
import tensorflow as tf
#import tf.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D, Input, Reshape, add, DepthwiseConv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np
import os
import logging
import sys
import io
from MAIN import Conversions, COMPRESSION

def activations_compression(model, data, settings):
    # Get Output Activations
    x_test = data[2]
    batch_size = settings[0]
    number_of_layers = model.layers
    print('number of layers:', len(number_of_layers))
    num_layers = len(number_of_layers)
    all_layers = list()
    for layer_index in range(1, num_layers):
        all_layers.append(model.get_layer(name=None, index=layer_index).output)
        #    print('intermediate layer number', layer_index, 'is layer:', model.get_layer(name=None, index=layer_index).output)
        #print('intermediate layer activations:', Model(inputs=model.input, outputs=model.get_layer(name=None, index=layer_index).output))

    intermediate_layer_model_input = model.input
    intermediate_layer_model = Model(inputs=intermediate_layer_model_input, outputs=all_layers)

    data = x_test
    num_batches = data.shape[0] // batch_size
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        intermediate_output = intermediate_layer_model.predict(data[start:end])
        #print("Intermediate result batch {}/{} done".format(batch_idx, num_batches))

    # loop to measure size of each layer, compresses it with all the compression algorithms and saves the ratios as well
    for i in range(len(intermediate_output)):
        #adjust to save matrizes - which type of file and how?
        #with io.open("ResNetv2_activations_of_layer_" + str(i + 1) + ".txt", 'w', encoding='utf-8') as f:
            #f.write(str(intermediate_output[i]))
        #number of entries in matrix:
        num_entries = len(intermediate_output[i].flatten())
        print('number of entries of', i+1, 'layer is:', num_entries)
        #size of matrix in KB
        size_entries = Conversions.get_obj_size(intermediate_output[i])
        print('size of entries of', i+1, 'layer is: ', size_entries, 'KB')
        #compress with all the algorithms (try with huffman then add others)
        compression = COMPRESSION.compress_all(intermediate_output[i])
        #print(i + 1, 'th layer activations', intermediate_output[i])
        i += 1
    return


#save the activation & weight matrizes from each individual layer as well as the global matrix
#try epoch later maybe
def activations_weights(model, data, settings, modelname):
    #activations

    x_test = data[2]
    batch_size = settings[0]
    number_of_layers = model.layers
    num_layers = len(number_of_layers)
    print('number of layers:', len(number_of_layers))
    all_layers = list()
    for layer_index in range(1, num_layers):
        all_layers.append(model.get_layer(name=None, index=layer_index).output)
        #    print('intermediate layer number', layer_index, 'is layer:', model.get_layer(name=None, index=layer_index).output)
        #print('intermediate layer activations:', Model(inputs=model.input, outputs=model.get_layer(name=None, index=layer_index).output))

    intermediate_layer_model_input = model.input
    intermediate_layer_model = Model(inputs=intermediate_layer_model_input, outputs=all_layers)

    data = x_test
    num_batches = data.shape[0] // batch_size
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        intermediate_output = intermediate_layer_model.predict(data[start:end])
        #print("Intermediate result batch {}/{} done".format(batch_idx, num_batches))

    #check this and change https://careerkarma.com/blog/python-typeerror-list-object-is-not-callable/#:~:text=Conclusion-,The%20Python%20%E2%80%9Ctypeerror%3A%20'list'%20object%20is%20not%20callable,list%20rather%20than%20curly%20brackets.
    global_a = intermediate_output[0].flatten()
    print('length of activations layers:', len(intermediate_output))
    for i in range(len(intermediate_output)):
        a = intermediate_output[i]
        # save activations as np.arrays to according file
        np.save(modelname + '_activations_' + str(i + 1), a)
        if i > 0:
            a_g = a.flatten()
            global_a = np.concatenate((global_a, a_g), axis = 0)
    np.save(modelname + 'global_activations', global_a)

    ######################################
    #weights
    global_weights = model.get_weights()
    #print('Global weights matrix:', global_weights)
    np.save(modelname + 'global_weights', global_weights)

    #https://ai-pool.com/d/how-to-get-the-weights-of-keras-model- to check loope
    i = 1
    for layer in model.layers:
        #print('i is: ', i)
        w = layer.get_weights()
        np.save(modelname + '_weights_' + str(i), w)
        print('Layer %s has weights of shape %s ' % (i, np.shape(w)))
        i += 1