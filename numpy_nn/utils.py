import numpy as np


def softmax(data):
    data = data - np.max(data, axis=1, keepdims=True)
    _exp = np.exp(data)
    _sum = np.sum(_exp, axis=1, keepdims=True)
    sm = _exp / _sum
    return sm


def sign(data):
    """
    Sign function for Perceptron
    sign(data) = 1 if data >= 0, -1 otherwise.

    [Inputs]
        data : input for sign function in any shape
    [Outputs]
        sign_data : sign value for data

    """
    sign_data = data.copy()
    sign_data[sign_data >= 0] = 1
    sign_data[sign_data < 0] = -1

    return sign_data
