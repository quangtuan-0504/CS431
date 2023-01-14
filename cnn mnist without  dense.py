import numpy as np
import matplotlib.pyplot as plt
import math
import random
from keras.layers import Dense, Input,Flatten,Conv2D,MaxPooling2D
from keras import Model,models,optimizers
import tensorflow as tf
from sklearn import preprocessing
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from numpy import newaxis

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("shape x,y_train:",x_train.shape,y_train.shape)
print("shape x,y_test:",x_test.shape,y_test.shape)
# reshape anh ve vector moi su dung dc cho  artifical neural network
x_train = x_train / 255
x_test = x_test / 255


X_train=x_train.reshape(60000,28,28,1)# có cx dc ko có cx dc, lớp Input tự reshape
print(X_train.shape)
X_test=x_test.reshape(10000,28,28,1)
#Y_train=y_train.reshape(len(y_train),1)
#print(Y_train.shape)
#y_train=y_train[:,newaxis]
#bỏ activate, pooling,dense, bo ca 3
# so sanh, ke bang co các thuộc tính accurate, time, para để ss
class cnn:
    def __init__(self):
        return None
    def build(self):
        self.model = tf.keras.Sequential([
            Conv2D(32, (3, 3),input_shape=(28,28,1)),#khi dùng cái này thì nó ko tự reshape về 28,28,1 dc mà mình phải tự reshape
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3)),
            MaxPooling2D((2, 2)),
            #Conv2D(10, (5, 5)),
            Conv2D(64, (3, 3)),#thay cái trên bằng cái này cx chạy dc
            Flatten()
            ])
        return self.model
    def train(self,x_train,y_train):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        hist=self.model.fit(x_train,y_train,epochs=2)
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
#bthg ta nghĩ 10 lớp thì out model 10 node r onehot y_train ra 10,
#out model là 64 node mà có 10 lớp nó lan truyền ngược kiểu j, thì ko phải, ta đag nhầm đấy, có 10 lớp chứ ko phải one hot 10, mà loss=sparsecategorical tự onehot ra 64 để lan truyền ngược
#BỞI HÀM LOSS=SPARSE_CateGo TỰ ĐỘNG ONEHOT RA VECTOR CÓ SHPAPE GIỐNG OUTPUT CỦA MODEL, nếu dùng loss= categorical thì ta phải onehot y_train ra vector cùng shape với out của model
Cnn_Model=cnn()
Cnn_Model.build()

Cnn_Model.summary()

start_time=time.time()# lấy thời điểm hiện tại lm mốc bắt đầu
hist=Cnn_Model.train(X_train,y_train)#ở model tạo I/O như nào thì ở đây mỗi điểm x,y_train phải cùng dạng với I/O
#https://stackoverflow.com/questions/63279168/valueerror-input-0-of-layer-sequential-is-incompatible-with-the-layer-expect
fn_time=time.time()#lấy thời điểm hiện tại lm mốc kết thúc

#thời gian huấn luyện
print("time_train:",fn_time-start_time)


#độ chính xác
y_pred_probability=Cnn_Model.predict(X_test)
print(y_pred_probability[0])
y_pred=[y.argmax() for y in y_pred_probability]
print(y_pred[0:5])
print(y_test[0:5])

print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))



cnf_matrix = confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
class_names=[0,1,2,3,4,5,6,7,8,9]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

Cnn_Model.summary()