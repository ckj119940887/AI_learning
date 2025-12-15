import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from common.gradient import numerical_diff

# f(x) = 0.01x^2 + 0.1x
def f(x):
    return 0.01 * x ** 2 + 0.1 * x

# 切线方程：f(x) = ax+b
def tangent_function(f, x):
    a = numerical_diff(f, x)

    b = f(x) - a * x
    return lambda x: a * x + b

x = np.arange(0.0, 20.0, 0.1)
y = f(x)

tf = tangent_function(f, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
