"""
first implement deep neural networks with numpy, no framework at all
if interested in machine learning, the scikit-learn docs actually make for great notes
"""

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3 * x ** 2 - 4 * x + 5

def f1(x):
    return 2 * x ** 2 - 10 * x + 5

def derivative(_f, _x, _h=0.000001):
    return (_f(_x + _h) - _f(_x)) / _h

def main():
    xs = np.arange(-10, 10, 0.25)

    ys1 = f1(xs)
    ds1 = derivative(f1, xs)
    plt.plot(xs, ys1)
    plt.plot(xs, ds1)

    plt.show()



if __name__ == '__main__':
    main()