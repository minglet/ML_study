import sys
sys.path.append('..')
from common.np import *

class SGD:
    '''
    W <- W - lr * (dL/dW)
    '''

    def __init__(self, lr=0.01) -> None:
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * (grads[i])