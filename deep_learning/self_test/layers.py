import numpy as np
from functions import softmax, cross_entropy_error
'''
Affine()
Sigmoid()
SoftmaxWithLoss()
'''

class Affine:
    '''
    L = W * x + b
    '''
    # 외부에서 전달되는 값을 넣어줌
    # 목적은 params와 grads를 업데이트 시켜주는 것
    def __init__(self, W, b) -> None:
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    # 외부에서 x를 전달해줄 것임
    def forward(self, x):
        W, b = self.params
        out = np.matmul(W, x) + b
        # x update
        self.x = x
        return out

    # 외부에서 dout을 전달해줄 것임
    '''
    dW = x.T*dout
    dx = dout*W.T
    db = sum(dout)
    '''
    def backward(self, dout):
        W, b = self.params
        dW = np.matmul(self.x.T, dout) 
        dx = np.matmul(dout, self.W.T)
        db = np.sum(dout, axis=0)

        # 깊은 복사
        self.grads[0][...] = dW
        self.grads[1][...] = db
        
        return dx

class Sigmoid:
    '''
    1 / (1 + exp(-x))
    '''
    def __init__(self) -> None:
        self.params = []
        self.grads = []
        self.x = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    # params, grads는 업데이트 시키지 않음
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
class SoftmaxWithLoss:
    # 외부에서 주어지는 값 없음
    def __init__(self) -> None:
        self.params = []
        self.grads = []
        self.t = None
        self.y = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        
        loss = cross_entropy_error(self.y, self.t)
        return loss
    
    def backward(self, dout):
        # 잘 이해가 가지 않음 데이터 넣어서 직접 print 해보기
        # t 실제 값, y 예측 값
        batch_size = self.t.shape[0]
                
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx /= batch_size
        
        return dx
