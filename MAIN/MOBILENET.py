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
from MAIN import QUANTIZER
from MAIN import Settings

#blocks & functions
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
    # return Activation(relu6, name=basename + '_activation')(x)
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

#shape (224, 224, 3) = input_shape
def MobileNet(data, settings, width_multiplier=1.0, q=None):
    model_type = 'MobileNetv2'
    batch_size = settings[0]
    epochs = settings[1]
    data_augmentation = settings[2]
    weight_precision = settings[5]
    activations_prescision = settings[6]
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
    if q == True:
        model = QUANTIZER.apply_quantization(model, pruning_policy=None, weight_precision=weight_precision, activation_precision=activations_prescision,
                                             activation_margin=None)
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
              optimizer=Adam(learning_rate=Settings.lr_schedule(0)),
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


    lr_scheduler = LearningRateScheduler(Settings.lr_schedule)

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
    '''
    data = x_test
    num_batches = data.shape[0] // batch_size
    for batch_idx in range(num_batches):
       start = batch_idx * batch_size
       end = start + batch_size
       intermediate_output = intermediate_layer_model.predict(data[start:end])
       print("Intermediate result batch {}/{} done".format(batch_idx, num_batches))

    print("Got intermediate, a random entry: {}".format(intermediate_output[0][0]))
    '''
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    #model.save("mobilenet_model.h5")
    return model
