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
from MAIN import Settings, QUANTIZER, Blocks
import io

def VGG19(data, settings, q):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    batch_size = settings[0]
    epochs = settings[1]
    data_augmentation = settings[2]
    weights_precision = settings[5]
    activations_prescision = settings[6]
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    y_test = data[3]
    x_train_mean = data[4]
    input_shape = data[5]
    num_classes = data[6]
    depth = 26

    model = Sequential()
    # weight_decay = self.weight_decay  #change weights??
    # every first model.add at end of bracets: kernel_regularizer=regularizers.l2(weight_decay)

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Instantiate model.
    #    model = Model(inputs=inputs, outputs=outputs)
    model_type = 'VGG19'
    # Quantize
    # a/w: 8/8, 16/16, 16/8
    if q == True:
        model = QUANTIZER.apply_quantization(model, pruning_policy=None, weight_precision=weights_precision, activation_precision=activations_prescision,
                                             activation_margin=None)
    num_layers = 60
    all_layers = list()
    for layer_index in range(1, num_layers):
        all_layers.append(model.get_layer(name=None, index=layer_index).output)
        #    print('intermediate layer number', layer_index, 'is layer:', model.get_layer(name=None, index=layer_index).output)
        print('intermediate layer activations:',
              Model(inputs=model.input, outputs=model.get_layer(name=None, index=layer_index).output))

    intermediate_layer_model_input = model.input
    intermediate_layer_model = Model(inputs=intermediate_layer_model_input, outputs=all_layers)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=Settings.lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()
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