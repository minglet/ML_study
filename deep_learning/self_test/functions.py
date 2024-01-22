import numpy as np

def sigmoid(x):
    return 1/1+np.exp(-x)

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True) 
    else:
        x = x - np.max(x) # x를 max(x)로 먼저 빼주고 x를 다시 정의해준다.
        x = np.exp(x)
        x /= np.sum(x)
    return x

def cross_entropy_error(y, t):
    if t.ndim == 1:
        t.reshape = (1, t.size) # 세로로 줄세운다고 생각하면 됨
        y.reshape = (1, y.size)
    
    if t.size == y.size:
        t = t.argmax(axis=1)

    # batch_size 정의
    batch_size = y.shape[0] # 차원에 맞춰서 batch_size가 결정

    # t는 k번째 클래스에 해당하는 정답레이블로 원-핫 벡터로 표현. 
    # 즉 1에 해당하는 값만 -np.log(요 안에 들어감)
    cross_entropy = np.log(y[np.arange(batch_size), t] + 1e-7) # log가 inf나지 않도록 작은 값을 더해줌
    loss = -np.sum(cross_entropy)

    return loss