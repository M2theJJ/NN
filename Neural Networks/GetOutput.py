from keras.models import Model

model = ...  # include here your original model

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)



#Get Output @ end try GetOutput.py
from keras.models import Model

#model = resnet_v1(input_shape=input_shape, depth=depth)  # include here your original model
#model = resnet_v1(input_shape, depth, num_classes=10)  # include here your original model

num_layers = 20
all_layers = list()
for layer_index in range(num_layers):
        all_layers.append(model.get_layer(layer_index).output

intermediate_layer_model = Model(inputs=model.input, outputs=all_layers)
intermediate_output = intermediate_layer_model.predict(data)
print('Outputs:', intermediate_output)