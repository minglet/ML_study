import numpy as np

def softmax(x):
    # ndim 2이상일땐 np.exp np.sum이렇게 쓰지않고 x.exp(axis=기준) 이렇게 사용
    if x.ndim == 2:
        x_max = x.max(axis=1, keepdims=True)
        return np.exp(x-x_max) / np.exp(x-x_max).sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x_max = np.max(x)
        return np.exp(x-x_max)/np.sum(np.exp(x-x_max))

def cross_entropy_error(y, t):
    # y는 정답레이블, t 예측 확률
    if y.ndim == 1:
        y.reshape(1, y.size) # y.shape (1, y.size)
        t.reshape(1, t.size) # t.shape (1, t.size)
    
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0] # shape과 size의 차이 shape은 (1, y.size) 이렇게 나옴
    # cross entropy error 식 

    cross_entropy = np.log(y[np.arange(batch_size), t]+ 1e-7)
    loss = -np.sum(cross_entropy)/batch_size
    return loss 