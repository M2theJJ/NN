'''import all the Models, Compressions & rest'''
#to run on terminal: python MAIN/RUN.py
#update git: git pull https://github.com/M2theJJ/NN
from __future__ import print_function
import sys
sys.path.append(".")
from MAIN import RESNET, VGG, MOBILENET
from MAIN import Settings, Conversions, QUANTIZER, W_A


'b_s = 32, epochs = 200, data_augemtation = true, num_classes = 10 (CIFAR10), substract_pixel_mean = true'
settings = Settings.training_parameters(32, 200, True, 10, True, None, None) #returns bs, ep, da, nc, spm, qwp, qap
data = Settings.data(10, settings)  #returns x_train, y_train, x_test, y_test, x_train_mean, input_shape, dataset
#resnet1 = RESNET.ResNet(1, 3, data, settings, q=False)
#W_A.activations_weights(resnet2, data, settings, "Resnetv1")
#resnet2 = RESNET.ResNet(2, 3, data, settings, q=False)
#W_A.activations_weights(resnet2, data, settings, "Resnetv2")
vgg19 = VGG.VGG19(data, settings, q=False)
W_A.activations_weights(vgg19, data, settings, "VGG19_10")
#mobilenet = MOBILENET.MobileNet(data, settings, width_multiplier=1.0, q=False)
#W_A.activations_weights(mobilenet, data, settings, "MobileNet")
#loaded_model = load_model("resnetv2_model.h5")
#compression = activations_compression(resnet2, data, settings)
#activations_weights(resnet2, data, settings, "ResNetv2")
#activations_weights(loaded_model, data, settings, "ResNetv2")
