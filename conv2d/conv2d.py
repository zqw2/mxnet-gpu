from mxnet import autograd, nd
from mxnet.gluon import nn


####################
# 实现互相关运算
####################
def corr2d(X, K): # 本函数已保存在d2lzh包中⽅便以后使⽤
    # X是个二维数据，并非四维
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = nd.array([[0, 1], [2, 3]])
print(corr2d(X, K))


#####################
# 设计卷积核进行边缘检测
#####################
X = nd.ones((6, 8))
X[:, 2:6] = 0
print(X)
K = nd.array([[1, -1]])
Y = corr2d(X, K)
print(Y)

#####################
# 二维卷积层
#####################
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))
    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()

#构造⼀个输出通道数为1（将在“多输⼊通道和多输出通道”⼀节介绍通道），核数组形状是(1, 2)的⼆
# 维卷积层
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()
# ⼆维卷积层使⽤4维输⼊输出，格式为(样本, 通道, ⾼, 宽)，这⾥批量⼤⼩（批量中的样本数）和通
# 道数均为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # 简单起⻅，这⾥忽略了偏差
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    # if (i + 1) % 2 == 0:
    print('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))