#https://www.youtube.com/watch?v=ho6JXE3EbZ8
import tensorflow as tf
from keras.applications.vgg16 import VGG16,decode_predictions
from keras.layers import Input
from keras import Model
import cv2
from numpy import newaxis
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

#load model train sẵn test dự đoán 1 ảnh
#lấy dc các filter của lớp conv đầu tiên trog model, visualize filter
#cắt model lớn ra 1 model nhỏ mới sau đó cho 1 bức ảnh qua model nhỏ đó để tạo ra 1 feature map đầu ra
# visualize feature map

model=VGG16(include_top=True,weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
model.summary()


#cách lấy layers
#https://www.tensorflow.org/api_docs/python/tf/keras/Model#get_layer
Layers=model.layers#tập hợp các layer của model
#print(layers)
filters,bias=Layers[1].get_weights()#
print(Layers[1].name)#block1_conv1
print(Layers[3].name)
print(filters.shape,bias.shape)#(3, 3, 3, 64) (64,)
#plot filters

fig1=plt.figure(figsize=(8, 12))#https://www.geeksforgeeks.org/matplotlib-pyplot-figure-in-python/
columns = 8
rows = 8
n_filters = columns * rows
for i in range(1, n_filters +1):# plot từng filter
    f = filters[:, :, :, i-1]
    fig1 =plt.subplot(rows, columns, i)
    fig1.set_xticks([])  #Turn off axis
    fig1.set_yticks([1,2])
    plt.imshow(f[:, :, :], cmap='gray') #Show only the filters from 0th channel (R)
    #ix += 1

plt.show()


#### Now plot filter outputs

#conv_layer_index = [1, 2, 4, 5, 7, 8, 9,...]
conv_layer_index = [1,2,4,8]
outputs = [model.layers[i].output for i in conv_layer_index]
#outputs=model.layers[5].output
print(outputs)
# mục đích là muốn lấy feature map của từng conv layers 1,2,4
model_short = Model(inputs=model.inputs, outputs=outputs)#model này có 3 output, lần lượt của conv layer 1,2,4
#thay vì việc tạo 3 model_short r lấy 3 output, thì tạo 1 model r lấy 3 output ở từng lớp, cách làm như trên
# nối input, và list các layers của model vgg16 từ layer[1] đến cái layer cuối trong outputs
print(model_short.summary())

#Input shape to the model is 224 x 224. SO resize input image to this shape.
from keras.preprocessing.image import load_img, img_to_array

img_path = "images/goldfish.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
#VGG user 224 as input

# convert the image to an array
img = img_to_array(img)
# expand dimensions to match the shape of model input
img = np.expand_dims(img, axis=0)

# Generate feature output by predicting on the input image
feature_output = model_short.predict(img)
print("feature ouput:",len(feature_output))


print("feature ouput[0]:",feature_output[0].shape)
columns = 8
rows = 8
for ftr in feature_output:
    #pos = 1
    fig=plt.figure(figsize=(12, 12))
    for i in range(1, columns*rows +1):
        fig =plt.subplot(rows, columns, i)
        fig.set_xticks([])  #Turn off axis
        fig.set_yticks([i])
        plt.imshow(ftr[0, :, :, i-1], cmap='gray')
        #pos += 1
plt.show()



