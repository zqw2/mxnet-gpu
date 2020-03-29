import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

print(mx.cpu(), mx.gpu(), mx.gpu(1), mx.gpu(2))