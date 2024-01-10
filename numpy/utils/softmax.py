import numpy as np

def softmax(x, dim=None):
    # x - np.max(x, axis=dim, keepdims=True)는 지수 함수의 지수가 너무 크거나 작아지는 것을 방지하기 위함
    # keepdims=True는 차원을 유지하겠다는 의미
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / e_x.sum(axis=dim, keepdims=True)