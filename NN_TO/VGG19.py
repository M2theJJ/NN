import tensorflow as tf
import tensorflow.keras,os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D , Flatten, Activation, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
import logging
import sys
from Extras import RAS
from Extras import Try




num_classes = 10
depth = 26
model_type = 'VGG19%dv%d'
subtract_pixel_mean = True
data_augmentation = True
batch_size = 64 #32
epochs = 1
#--------------------------------------------------------------------
#import library and resize images

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

#test input:
#x_train = tf.ones((50000, 32, 32, 3))

#------------------------------------------------------------------------------
def lr_schedule(epoch):

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

#Build Model
def VGG19():
# Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    model = Sequential()
    #weight_decay = self.weight_decay  #change weights??
    #every first model.add at end of bracets: kernel_regularizer=regularizers.l2(weight_decay)

    model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=input_shape))
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

    return model

model = VGG19()
#--------------------------------------------------------------------------------
#Get Output Activations
num_layers = 60
all_layers = list()
for layer_index in range(1, num_layers):
    all_layers.append(model.get_layer(name=None, index=layer_index).output)
#    print('intermediate layer number', layer_index, 'is layer:', model.get_layer(name=None, index=layer_index).output)
    print('intermediate layer activations:', Model(inputs=model.input, outputs=model.get_layer(name=None, index=layer_index).output))

intermediate_layer_model_input = model.input
intermediate_layer_model = Model(inputs=intermediate_layer_model_input, outputs=all_layers)
#---------------------------------------------------------------------------------

#number of layers
number_of_layers = model.layers
print('The number of layers is:', len(number_of_layers))
'''
#--------------------------------------------------------------------------------
#Get Weights of Layer
for layer in model.layers:
    weights_l = layer.get_weights() # list of numpy arrays
    #print('weights', weights_l)

#Global Weights of Model
#model = VGG19 predefined
weights_g = model.get_weights() #returns numpy list
#print('weights', weights_g)

#write weights to file
filename = "weights_VGG19.txt"
#text = RAS.convert_f_array_to_file(weights_l, filename)
#---------------------------------------------------------------------------------
'''
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule(0)),
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
#logging

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

'''
act_first_layer = intermediate_output[0]
act_last_layer = intermediate_output[58]
for i in range(0, len(intermediate_output)):
    print(i,'th layer activations', intermediate_output[i])
    i+=1
#print('frist layer activations', act_first_layer)
#print('last layer activations', act_last_layer)
'''

'''
#within 6th dim the floats/activations are
print("Got intermediate, a random entry: {}".format(intermediate_output[0][0]))
print('Type of intermediate_layer_model:', type(intermediate_output), 'type of intermediate_layer_model entry', type(intermediate_output[0][0]))
print('#rows', len(intermediate_output), 'rows', intermediate_output) #59 - maybe layers? pretty sure layers
print('#colums', len(intermediate_output[0]), 'colums', intermediate_output[0]) #32
print('#depth', len(intermediate_output[0][0]), 'depth', intermediate_output[0][0]) #32
print('#4th dimension?', len(intermediate_output[0][0][0]), '4th dimension?', intermediate_output[0][0][0]) #32
print('#5th dimension?', len(intermediate_output[0][0][0][0]), '5th dimension?', intermediate_output[0][0][0][0]) #64 float array
print('6th dimension?', intermediate_output[0][0][0][0][0]) #actual float number
#print('#7th dimension?', len(intermediate_output[0][0][0][0][0][0]), '5th dimension?', len(intermediate_output[0][0][0][0][0][0])) #doesn't exist
'''
#write array to file
#filename = "activations_VGG19.txt"
#text = RAS.convert_f_array_to_file(intermediate_output[0][0][0][0], filename)
#creates txt files for each layer
import io
for i in range(len(intermediate_output)):
    with io.open("VGG19_activations_of_layer_" + str(i) + ".txt", 'w', encoding='utf-8') as f:
        f.write(str(intermediate_output[i]))
    print(i, 'th layer activations', intermediate_output[i])
    i+=1
'''
#slice array and write on file
s_filename = "sliced_activations_VGG19.txt"
sliced_arr = Try.slice_float_array(intermediate_output[0][0][0][0])
sliced_array = sliced_arr[0]
sliced_arr_l = sliced_arr[1]
sliced_arr = sliced_array.astype(int)
sliced_text = RAS.convert_int_array_to_file(sliced_array, s_filename)
'''
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

'''
rows = len(intermediate_output)
colums = len(intermediate_output[0])
i = 0
h = 0
#write activations into file
a_str = RAS.convert_list_to_string(intermediate_output)
filename = "activations_VGG.txt"
for i in range(0, rows):
    i += 1
    for h in range(0,colums):
        activations_VGG19_text = RAS.convert_f_array_to_file(intermediate_output[i][h], filename)
        h += 1

'''
# Redirect stdout and stderr
sys.stdout = LogFile('stdout')
sys.stderr = LogFile('stderr')




# 60 layers

