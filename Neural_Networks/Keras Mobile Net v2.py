import tensorflow as tf
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

tf.keras.applications.MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
)

num_layers = 20
all_layers = list()
for layer_index in range(1, num_layers):
    all_layers.append(model.get_layer(name=None, index=layer_index).output)

intermediate_layer_model_input = model.input
intermediate_layer_model = Model(inputs=intermediate_layer_model_input, outputs=all_layers)
