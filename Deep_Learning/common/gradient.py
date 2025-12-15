import numpy as np

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def _numerical_gradient(f, x):
    grad = np.zeros_like(x)

    h = 1e-4
    for i in range(x.size):
        tmp = x[i]
        # 只改一个
        x[i] = tmp + h
        fxh1 = f(x)

        x[i] = tmp - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2)/(2 * h)

        # 恢复原来的值
        x[i] = tmp

    return grad

# 扩展到多维矩阵
def numerical_gradient(f, x):
    if x.ndim == 1:
        return _numerical_gradient(f, x)
    else:
        grad = np.zeros_like(x)
        for i in range(x.size):
            grad[i] = numerical_gradient(f, x[i])