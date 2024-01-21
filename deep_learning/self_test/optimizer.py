
from collections.abc import Iterable


class SGD:
    '''
    W <- W - lr * (dL / dW)
    '''

    def __init__(self, lr=0.01) -> None:
        self.lr = lr
    
    def update(self, params, grads):
        # 모든 파라미터들을 업데이트 시키는 것이 목적!
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
    # 따로 return 값 없음