from keras.applications import VGG16
import matplotlib.pyplot as plt


model=VGG16(include_top=True,weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
model.summary()



