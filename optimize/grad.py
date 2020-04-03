import d2lzh as d2l
import math
from mxnet import nd
import numpy as np

def gd(eta):
    x = 10
    results = [x]
    for i in range(60):
        x -= eta * 2 * x # f(x) = x * x的导数为f'(x) = 2 * x
        results.append(x)
    print('epoch 10, x:', x)
    return results
res = gd(1)

def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    d2l.set_figsize()
    d2l.plt.plot(f_line, [x * x for x in f_line])
    d2l.plt.plot(res, [x * x for x in res], '-o')
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')
    d2l.plt.show()
show_trace(res)