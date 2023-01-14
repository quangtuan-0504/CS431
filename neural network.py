import numpy as np
import matplotlib.pyplot as plt
import math
import random
from keras.layers import Dense, Input
from keras import Model,models,optimizers
#b1 generate data
# co the dung du lieu co san trong scikit learn
#https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html#sphx-glr-auto-examples-decomposition-plot-kernel-pca-py
mean = [5,6]
cov = [[1, 0],[0,1]]
N = 5000
X0 = np.random.multivariate_normal(mean, cov, N)
print(X0.shape)
lst1=[]
lst2=[]
for x1,x2 in X0:
    norm=math.sqrt((x1-mean[0])**2+(x2-mean[1])**2)
    if norm<0.9:
        lst1.append([x1,x2])
    elif norm>1.4:
        lst2.append([x1,x2])
lst1=np.array(lst1)
lst2=np.array(lst2)
#print(lst2.shape)
X = np.concatenate((lst1,lst2), axis = 0)
print(X.shape)
Y= np.asarray([1]*lst1[:,0].size + [0]*lst2[:,0].size).T
print(Y.shape,lst1.size+lst2.size)
plt.scatter(lst1[:,0],lst1[:,1],c='#d62728')
plt.scatter(lst2[:,0],lst2[:,1],c='#9467bd')
plt.show()
#b2 clone code tu bai logistic
#b3 sua lai ham build voi 1 lop an, 5 neural
class neural_network:
    def __init__(self):
        return None
    def build(self,in_dim=2):
        input=Input(in_dim)
        hidden_layer=Dense(5,activation='relu',use_bias=True)(input)
        output=Dense(1,activation='sigmoid',use_bias=True)(hidden_layer)
        self.model=Model(input,output)
        return self.model
    def train(self,x_train,y_train):
        otm=optimizers.SGD(learning_rate=0.0001,momentum=0.9)
        self.model.compile(optimizer=otm,loss='binary_crossentropy')
        hist=self.model.fit(x_train,y_train,epochs=30000)
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


#b4 train va visualize loss
Neural_network_Model=neural_network()
Neural_network_Model.build()
hist=Neural_network_Model.train(X,Y)
weights=Neural_network_Model.get_paramater()
print(weights)

plt.plot(hist.history['loss'])
plt.savefig("nn1.png",bbox_inches="tight")
plt.show()

for i in range(0,5):
    w0 = weights[1][i]
    w1 = weights[0][0][i]  # x
    w2 = weights[0][1][i]  # y
    print(w0, w1, w2)
    x = np.arange(X[:,0].min(), X[:,0].max(), 1)
    y = -w1 / w2 * x - w0 / w2
    plt.plot(x,y)

plt.scatter(lst1[:,0],lst1[:,1],c='#d62728')
plt.scatter(lst2[:,0],lst2[:,1],c='#9467bd')
plt.savefig("nn2.png",bbox_inches="tight")
#plt.xlim(X[:,0].min(), X[:,0].max())
#plt.ylim(X[:,1].min(), X[:,1].max())
#plt.plot(x1,y1)
#plt.plot(x2,y2)
plt.show()



#b5 ve decision boundary
xm = np.arange(X[:,0].min(), X[:,0].max(), 0.015)
xlen = len(xm)
ym = np.arange(X[:,1].min(), X[:,1].max(), 0.015)
ylen = len(ym)
xx, yy = np.meshgrid(xm, ym)


xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)

print(xx.shape, yy.shape)
print(xx1.shape, yy1.shape)
XX = np.concatenate(( xx1, yy1), axis = 0)
print(XX.shape)
XX=XX.T
print(XX.shape)
Z=Neural_network_Model.predict(XX)
print(Z.shape)
Z1=[]
for z in Z:
    if z<0.5:
        Z1.append(0)
    else:
        Z1.append(1)
Z1=np.array(Z1)
Z1 = Z1.reshape(xx.shape)
print(Z1)

CS = plt.contourf(yy, xx, Z1, 200, cmap='jet', alpha = .1)


#plt.xlim(X[:,0].min(), X[:,0].max())
#plt.ylim(X[:,1].min(), X[:,1].max())
plt.xticks(())
plt.yticks(())

plt.scatter(lst1[:,0],lst1[:,1],c='#d62728')
plt.scatter(lst2[:,0],lst2[:,1],c='#9467bd')
plt.savefig("nn3.png",bbox_inches="tight")
plt.show()
