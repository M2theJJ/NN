from __future__ import print_function


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

#from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper as QuantizeWrapper


#from tensorflow.keras.utils.vis_utils import plot_model
from Compression_TO import Compression_Main
from Extras import RAS
from NN_TO import Quantizer_TO

#https://machinelearningmastery.com


def get_layer_output_model(model):
    num_layers = len(model.layers)
    all_layers = list()

    #we do not care about the output of layers as dropout or flatten
    layers_of_interest = [QuantizeWrapper, Dense, Conv2D, DepthwiseConv2D]

        #, Dense, Conv2D, DepthwiseConv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D]


    for layer_index in range(0, num_layers):
        layer = model.get_layer(name=None, index=layer_index)

        found = False
        for layer_type in layers_of_interest:
            if isinstance(layer, layer_type):
                all_layers.append(layer.input)
                all_layers.append(layer.output)
                print('Found layer for storing results {}, storing: {} and '.format(layer.name, layer.input.name,  layer.output.name ))
                found = True
                break

        if found is False:
            print("Not storing activation output for layer: {}".format(layer.name))

    intermediate_layer_model_input = model.get_layer(name=None, index=0).input
    print("Found model input: {} from layer {}".format(intermediate_layer_model_input.name,  model.get_layer(name=None, index=0).name))

    intermediate_layer_model = Model(inputs=intermediate_layer_model_input, outputs=all_layers)
    return intermediate_layer_model


#b_s = 32, epochs = 200, data_augemtation = true, num_classes = 10 (CIFAR10), substract_pixel_mean = true
def training_parameters(batch_size, epochs, data_augmentation, num_classes, substract_pixel_mean):
    bs = batch_size
    ep = epochs
    da = data_augmentation
    nc = num_classes
    spm = substract_pixel_mean
           #0 #1   #2  #3  #4
    return bs, ep, da, nc, spm

#loads Data - need to add other data load options ( & Imagenet!)
def data(dataset, settings):
    num_classes = settings[3]
    subtract_pixel_mean = settings[4]
    # Load the CIFAR10 data.
    if dataset == 10:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if dataset == 100:
        (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.cifar100.load_data(label_mode="fine")
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

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """

    # Model parameter
    # ----------------------------------------------------------------------------
    #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
    # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
    #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
    # ----------------------------------------------------------------------------
    # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
    # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
    # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
    # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
    # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
    # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
    # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
    # ---------------------------------------------------------------------------
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(n, data):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    depth = n * 6 + 2
    input_shape = data[5]
    num_classes = data[6]
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(n, data):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    depth = n * 9 + 2
    input_shape = data[5]
    num_classes = data[6]
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')

    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def _conv_block(inputs, filters, kernel, strides, use_bias=True):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if nlay < 0 or nlay > 16:
        basename = 'conv_%d' % (nlay + 1)
    else:
        basename = 'expanded_conv_%d_expand' % nlay

    x = Conv2D(filters, kernel, padding='same', strides=strides, name=basename, use_bias=use_bias)(inputs)
    x = BatchNormalization(axis=channel_axis, name=basename + '_batch_normalization')(x)
    #return Activation(relu6, name=basename + '_activation')(x)
    return tf.keras.layers.ReLU(6)(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """
    global nlay

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Create expansions layer only if needed (expansion factor >1)
    if t > 1:
        tchannel = K.int_shape(inputs)[channel_axis] * t
        x = _conv_block(inputs, tchannel, (1, 1), (1, 1), use_bias=False)
    else:
        x = inputs

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same',
                        name='expanded_conv_%d_depthwise' % nlay, use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis, name='expanded_conv_%d_depthwise_batch_normalization' % nlay)(x)
#    x = Activation(relu6, name='expanded_conv_%d_depthwise_activation' % nlay)(x)
    x = tf.keras.layers.ReLU(6)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='expanded_conv_%d_project' % nlay, use_bias=False)(
        x)
    x = BatchNormalization(axis=channel_axis, name='expanded_conv_%d_project_batch_normalization' % nlay)(x)

    if r:
        x = add([x, inputs], name="expanded_conv_%d_add" % nlay)

    nlay += 1
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x

def relu6(x):
    return tf.keras.layers.ReLU(6)

def roundup(n):
    x = (n + 6) // 8
    return x * 8

#n = 3
def ResNet(version, n, data, settings):
    batch_size = settings[0]
    epochs = settings[1]
    data_augmentation = settings[2]
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    y_test = data[3]
    x_train_mean = data[4]
    if version == 2:
        model = resnet_v2(n, data)
        depth = n * 9 + 2
    else:
        model = resnet_v1(n, data)
        depth = n * 6 + 2

    # Quantize
    # a/w: 8/8, 16/16, 16/8
    #model = Quantizer_TO.apply_quantization(model, pruning_policy=None, weight_precision=8, activation_precision=8,
                                                activation_margin=None)

    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, version)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()
    print(model_type)
    # number of layers
    number_of_layers = model.layers
    print('The number of layers is:', len(number_of_layers))

    # --------------------------------------------------------------------------------
    # Get Output Activations
    num_layers = len(number_of_layers)
    all_layers = list()
    for layer_index in range(1, num_layers):
        all_layers.append(model.get_layer(name=None, index=layer_index).output)
        #    print('intermediate layer number', layer_index, 'is layer:', model.get_layer(name=None, index=layer_index).output)
        #print('intermediate layer activations:', Model(inputs=model.input, outputs=model.get_layer(name=None, index=layer_index).output))

    intermediate_layer_model_input = model.input
    intermediate_layer_model = Model(inputs=intermediate_layer_model_input, outputs=all_layers)
    # ---------------------------------------------------------------------------------

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    print('filepath', filepath)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    #plot = plot_model(model)
    #see if install jupyter notebook to get image of plot.
    model.save("resnetv2_model.h5")
    return model

def VGG19(data, settings):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    batch_size = settings[0]
    epochs = settings[1]
    data_augmentation = settings[2]
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    y_test = data[3]
    x_train_mean = data[4]
    input_shape = data[5]
    num_classes = data[6]
    depth = 26

    model = Sequential(name="VGG19")
    # weight_decay = self.weight_decay  #change weights??
    # every first model.add at end of bracets: kernel_regularizer=regularizers.l2(weight_decay)
    #model.add(Input(shape=input_shape, name="input_layer"))
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    #model.add(BatchNormalization()) #wrong, vgg does not have batchnorm
    #model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Instantiate model.
    #    model = Model(inputs=inputs, outputs=outputs)
    model_type = 'VGG19'
    # Quantize
    # a/w: 8/8, 16/16, 16/8

    #model = Quantizer_TO.apply_quantization(model, pruning_policy=None, weight_precision=8, activation_precision=8, activation_margin=None)

    #

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()

    intermediate_layer_model = get_layer_output_model(model)

    print('model type', model_type)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)

    # ----------------------------------------------------------
    # logging

    class LogFile(object):
        """File-like object to log text using the `logging` module."""

        def __init__(self, name=None):
            self.logger = logging.getLogger(name)

        def write(self, msg, level=logging.INFO):
            self.logger.log(level, msg)

        def flush(self):
            for handler in self.logger.handlers:
                handler.flush()

    logging.basicConfig(level=logging.DEBUG, filename='../Keras_VGG19_CIFAR10.log')

    data = x_test
    num_batches = data.shape[0] // batch_size
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        intermediate_output = intermediate_layer_model.predict(data[start:end])
        print("Intermediate result batch {}/{} done".format(batch_idx, num_batches))

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # Redirect stdout and stderr
    sys.stdout = LogFile('stdout')
    sys.stderr = LogFile('stderr')
    model.save("VGG19_model.h5")
    return model



#shape (224, 224, 3) = input_shape
def MobileNet(data, settings, width_multiplier=1.0):
    model_type = 'MobileNetv2'
    batch_size = settings[0]
    epochs = settings[1]
    data_augmentation = settings[2]
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    y_test = data[3]
    x_train_mean = data[4]
    input_shape = data[5]
    num_classes = data[6]
    depth = 26
    k = settings[3]

    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
    # Returns
        MobileNetv2 model.
    """

    global nlay
    nlay = -1

    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, roundup(int(32 * width_multiplier)), (3, 3), strides=(2, 2), use_bias=False)
    nlay += 1

    fix = 0
    if width_multiplier - 1.3 < 0.01:
        fix = -2

    x = _inverted_residual_block(x, roundup(int(16 * width_multiplier)), (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, roundup(int(24 * width_multiplier)), (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, roundup(int(32 * width_multiplier)), (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, roundup(int(64 * width_multiplier) + fix), (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x, roundup(int(96 * width_multiplier)), (3, 3), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, roundup(int(160 * width_multiplier)), (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, roundup(int(320 * width_multiplier)), (3, 3), t=6, strides=1, n=1)

    last_conv_size = max(1280, int(1280 * width_multiplier))

    x = _conv_block(x, last_conv_size, (1, 1), strides=(1, 1), use_bias=False)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, last_conv_size))(x)
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(k, (1, 1), padding='same', name='logits', use_bias=True)(x)

    x = Activation('softmax', name='softmax')(x)
    outputs = Reshape((k,), name='out')(x)

    model = Model(inputs=inputs, outputs=outputs)

    #Quantize
    # a/w: 8/8, 16/16, 16/8
    #model = Quantizer_TO.apply_quantization(model, pruning_policy=None, weight_precision=8, activation_precision=8, activation_margin=None)

    #    plot_model(model, to_file='MobileNetv2.png', show_shapes=True)
    #--------------------------------------------------------------------------------
    #Get Output
    num_layers = 156
    all_layers = list()
    for layer_index in range(1, num_layers):
        all_layers.append(model.get_layer(name=None, index=layer_index).output)

    intermediate_layer_model_input = model.input
    intermediate_layer_model = Model(inputs=intermediate_layer_model_input, outputs=all_layers)
    #---------------------------------------------------------------------------------
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])
    model.summary()
    print(model_type)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)


    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True)


    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                    shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)



    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)


    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

    #----------------------------------------------------------
    data = x_test
    num_batches = data.shape[0] // batch_size
    for batch_idx in range(num_batches):
       start = batch_idx * batch_size
       end = start + batch_size
       intermediate_output = intermediate_layer_model.predict(data[start:end])
       print("Intermediate result batch {}/{} done".format(batch_idx, num_batches))

    print("Got intermediate, a random entry: {}".format(intermediate_output[0][0]))
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    model.save("mobilenet_model.h5")
    return model




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
        size_entries = RAS.get_obj_size(intermediate_output[i])
        print('size of entries of', i+1, 'layer is: ', size_entries, 'KB')
        #compress with all the algorithms (try with huffman then add others)
        compression = Compression_Main.compress_all(intermediate_output[i])
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



#################################################################
#b_s = 32, epochs = 200, data_augemtation = true, num_classes = 10 (CIFAR10), substract_pixel_mean = true
settings = training_parameters(32, 200, True, 100, True)
data = data(100, settings)
#resnet2 = ResNet(2, 3, data, settings)
vgg19 = VGG19(data, settings)
#activations_weights(vgg19, data, settings, "VGG19")
#mobilenet = MobileNet(data, settings)
activations_weights(vgg19, data, settings, "VGG19_100")
#loaded_model = load_model("resnetv2_model.h5")
#compression = activations_compression(resnet2, data, settings)
#activations_weights(resnet2, data, settings, "ResNetv2")
#activations_weights(loaded_model, data, settings, "ResNetv2")
