
from mxnet import init, nd
from mxnet.gluon import nn

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # 实际的初始化逻辑在此省略了

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
nn.Dense(10))
net.initialize(init=MyInit())


X = nd.random.uniform(shape=(1, 2))
Y = net(X)