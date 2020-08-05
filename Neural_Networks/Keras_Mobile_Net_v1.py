import tensorflow.keras
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import logging
import sys
from QuantizerD import Quantizer

depth = 16
version = 1

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

#num_classes for CIFAR_10 = 10
num_classes = 10


# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

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
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

#print('y_train', y_train)
#print('y_test', y_test)

def get_conv_block(tensor, channels, strides, alpha=1.0, name=''):
    channels = int(channels * alpha)

    x = Conv2D(channels,
               kernel_size=(3, 3),
               strides=strides,
               use_bias=False,
               padding='same',
               name='{}_conv'.format(name))(tensor)
    x = BatchNormalization(name='{}_bn'.format(name))(x)
    x = Activation('relu', name='{}_act'.format(name))(x)
    return x


def get_dw_sep_block(tensor, channels, strides, alpha=1.0, name=''):
    """Depthwise separable conv: A Depthwise conv followed by a Pointwise conv."""
    channels = int(channels * alpha)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        strides=strides,
                        use_bias=False,
                        padding='same',
                        name='{}_dw'.format(name))(tensor)
    x = BatchNormalization(name='{}_bn1'.format(name))(x)
    x = Activation('relu', name='{}_act1'.format(name))(x)

    # Pointwise
    x = Conv2D(channels,
               kernel_size=(1, 1),
               strides=(1, 1),
               use_bias=False,
               padding='same',
               name='{}_pw'.format(name))(x)
    x = BatchNormalization(name='{}_bn2'.format(name))(x)
    x = Activation('relu', name='{}_act2'.format(name))(x)
    return x


def MobileNet(shape, num_classes, alpha=1.0, include_top=True, weights=None):
    x_in = Input(shape=shape)

    x = get_conv_block(x_in, 32, (2, 2), alpha=alpha, name='initial')

    layers = [
        (64, (1, 1)),
        (128, (2, 2)),
        (128, (1, 1)),
        (256, (2, 2)),
        (256, (1, 1)),
        (512, (2, 2)),
        *[(512, (1, 1)) for _ in range(5)],
        (1024, (2, 2)),
        (1024, (2, 2))
    ]

    for i, (channels, strides) in enumerate(layers):
        x = get_dw_sep_block(x, channels, strides, alpha=alpha, name='block{}'.format(i))

    if include_top:
        x = GlobalAvgPool2D(name='global_avg')(x)
        x = Dense(num_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=x_in, outputs=x)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model

