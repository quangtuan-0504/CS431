import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras import Model,models
import random

class linear_regression:
    def __init__(self):
        return None
    def build(self,in_dim=1):
        input=Input(in_dim)
        output=Dense(1,use_bias=True)(input)
        self.model=Model(input,output)
        return self.model
    def train(self,x_train,y_train):
        self.model.compile(optimizer='SGD',loss='mse')
        hist=self.model.fit(x_train,y_train,epochs=1000)
        return hist
    def save(self,path_model):
        return self.model.save(path_model)
    def load(self,path_model):
        return models.load_model(path_model)
    def predict(self,x_test):
        return self.model.predict(x_test)
    def summary(self):
        return self.model.summary()
    def get_paramater(self):
        return self.model.get_weights()



#b1 generate data
x=np.arange(3,7,0.5)
y=-5*x+8+np.random.randn(len(x))
#x_train=np.concatenate(([np.ones(len(x))],[x]),axis=0)
print('X=',x)
print('y=',y)

#b2 build model
linearModel=linear_regression()
#b3 khoi tao model va train
linearModel.build(1)
hist=linearModel.train(x,y)
#b4 visualize_lossfunction_hyperplane

plt.plot(hist.history['loss'])
plt.show()

weight=linearModel.get_paramater()
print(weight)
linearModel.summary()
linearModel.save('mymodel')
y_pre=linearModel.predict(x)
print('Y_PRED:',y_pre)
model2=linearModel.load('mymodel')



model2.summary()

plt.plot(x,y,'ro')
plt.plot(x,y_pre)
plt.show()