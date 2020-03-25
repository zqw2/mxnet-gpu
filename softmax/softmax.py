import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time
from d2lzh import *

mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)


########################
#  数据集长度
########################
a =len(mnist_train)
b = len(mnist_test)
print(a, b)

########################
#  数据集feature,label
########################
feature, label = mnist_train[0]
print(feature.shape, feature.dtype)
print(label, type(label), label.dtype)


########################
#  数据集0-9图形化展示
########################
X, y = mnist_train[0:9]
show_fashion_mnist(X, get_fashion_mnist_labels(y))

########################
#  数据集多进程读取
########################
batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    num_workers = 0 # 0表⽰不⽤额外的进程来加速读取数据
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True, num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size, shuffle=False, num_workers=num_workers)

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))


########################
#  测试softmax运算
########################
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition # 这⾥应⽤了⼴播机制

X = nd.random.normal(shape=(2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(axis=1))