from mxnet import nd
from time import time
a = nd.ones(shape=1000)
b = nd.ones(shape=1000)
start = time()
c = nd.ones(shape=1000)
for i in range(1000):
    c[i] = a[i] + b[i]

print(time() - start)

start2 = time()

d = a + b

print(time() - start2)