from mxnet import nd
from mxnet.gluon import nn
# 定义⼀个函数来计算卷积层。它初始化卷积层权重，并对输⼊和输出做相应的升维和降维

def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1)代表批量⼤⼩和通道数（“多输⼊通道和多输出通道”⼀节将介绍）均为1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:]) # 排除不关⼼的前两维：批量和通道

# 注意这⾥是两侧分别填充1⾏或列，所以在两侧⼀共填充2⾏或列
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = nd.random.uniform(shape=(8, 8))

Y = comp_conv2d(conv2d, X)
print(Y.shape)
print(Y)
