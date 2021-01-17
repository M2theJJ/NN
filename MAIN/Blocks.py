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

#Get Output
def intermediate_layer_model(model):
    num_layers = len(model.layers)
    all_layers = list()
    for layer_index in range(1, num_layers):
        all_layers.append(model.get_layer(name=None, index=layer_index).output)

    intermediate_layer_model_input = model.input
    intermediate_layer_model = Model(inputs=intermediate_layer_model_input, outputs=all_layers)
    return intermediate_layer_model

def LOG(modelname):
    class LogFile(object):
        """File-like object to log text using the `logging` module."""

        def __init__(self, name=None):
            self.logger = logging.getLogger(name)

        def write(self, msg, level=logging.INFO):
            self.logger.log(level, msg)

        def flush(self):
            for handler in self.logger.handlers:
                handler.flush()

    logging.basicConfig(level=logging.DEBUG, filename='../'+str(modelname)+'.log')
