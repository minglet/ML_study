
class SGD:
    '''
    W <- W - lr*(dL/dW)
    '''
    def __init__(self, lr=0.01) -> None:
        self.lr = lr

    def update(self, greds, params):
        for i in range(len(params)):
            params -= self.lr * (greds[i])