from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.datasets import cifar10, cifar100
import numpy as np
import os
import logging
import sys
import io
from keras.utils.vis_utils import plot_model
from Compression_TO import Compression_Main
from Extras import RAS



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
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()


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
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
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
            x = keras.layers.add([x, y])
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
            x = keras.layers.add([x, y])

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
    #get weights of each layer

    #w_d = dict()
    #for layer in model.layers:
    #    print(layer.get_config(), layer.get_weights())
    #    layer_[layer]_weights = layer.get_weights(layer)
    #get global weights
    '''
####weights
#    weight_dict = {}
    global_weights = model.get_weights()
    print('Global weights matrix:', global_weights)
    np.save(modelname+'global_weights', global_weights)

    for layer_i in range(0,len(model.layers)-1):
        w = model.layers[layer_i].get_weights()[0]
        #saves weights as np.arrays to acording file
        np.save(modelname+'_weights_'+str(layer_i+1), w)
        print('Layer %s has weights of shape %s ' % (layer_i, np.shape(w)))
##############################################################
        # save all weights and biases inside a dictionary
        if epoch == 0:
            # create array to hold weights
            weight_dict['w_' + str(layer_i + 1)] = w

        else:
            # append new weights to previously-created weights array
            weight_dict['w_' + str(layer_i + 1)] = np.dstack(
                (weight_dict['w_' + str(layer_i + 1)], w))
    print('weights dictionary:', weight_dict)
    '''
####activations
# Get Output Activations
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
    #global_activations = intermediate_output
    #print('Global activations matrix:', global_activations)
    #np.save(modelname + 'global_activations', global_activations)
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
    #tf.keras.layers.Layer.get_weights()
    global_weights = model.get_weights()
    #print('Global weights matrix:', global_weights)
    np.save(modelname + 'global_weights', global_weights)

    '''
    i = 1
    for layer in model.layers:
        weights = layer.get_weights()
        print('weights'+ str(i) , weights)
        i += 1
    '''

    i = 1
    for layer in model.layers:
        #print('i is: ', i)
        w = layer.get_weights()
        #print('weights'+ str(i) , weights)
        #w = model.get_layer(index=i).get_weights()[0]
        #w = model.get_layers[i].get_weights()[0]
        #w = model.layers[i].get_weights()[0]
        # saves weights as np.arrays to acording file
        np.save(modelname + '_weights_' + str(i), w)
        print('Layer %s has weights of shape %s ' % (i, np.shape(w)))
        i += 1



#################################################################
settings = training_parameters(32, 1, True, 10, True)
data = data(10, settings)
#resnet2 = ResNet(2, 3, data, settings)
loaded_model = load_model("resnetv2_model.h5")
#compression = activations_compression(resnet2, data, settings)
#activations_weights(resnet2, data, settings, "ResNetv2")
activations_weights(loaded_model, data, settings, "ResNetv2")
