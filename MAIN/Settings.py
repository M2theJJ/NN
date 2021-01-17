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
#from tensorflow.keras.utils.vis_utils import plot_model

#b_s = 32, epochs = 200, data_augemtation = true, num_classes = 10 (CIFAR10), substract_pixel_mean = true
def training_parameters(batch_size, epochs, data_augmentation, num_classes, substract_pixel_mean, weights_precision, activations_precision):
    bs = batch_size
    ep = epochs
    da = data_augmentation
    nc = num_classes
    spm = substract_pixel_mean
    qwp = weights_precision
    qap = activations_precision
           #0 #1   #2  #3  #4   #5  #6
    return bs, ep, da, nc, spm, qwp, qap

#loads Data - need to add other data load options ( & Imagenet!)
def data(dataset, settings):
    num_classes = settings[3]
    subtract_pixel_mean = settings[4]
    # Load the CIFAR10 data.
    if dataset == 10:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if dataset == 100:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
        #tf.keras.datasets.cifar100.load_data(label_mode="fine")


    # Input image dimensions.
    input_shape = x_train.shape[1:]
    print('INPUTSHAPE:', input_shape)

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
            #0      #1       #2      #3      #4            #5           #6
    return x_train, y_train, x_test, y_test, x_train_mean, input_shape, dataset

#learning rate, input the parameters function
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """

    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr