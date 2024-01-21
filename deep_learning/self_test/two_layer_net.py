import numpy as np 
from layers import Affine, Sigmoid, SoftmaxWithLoss

class TwoLayersNet:
    def __init__(self, input_size, hidden_size, output_size) -> None:
        I, H, O = input_size, hidden_size, output_size

        W1 = 0.01 * np.random.randn(I, H)  # I,H shape의 평균 0, 표준편차 1의 랜덤한 값으로 채운 np.ndarray 생성
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        self.layers = {
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        }
        self.loss_layer = SoftmaxWithLoss()

        # params와 grads를 모두 리스트에 저장
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x) # x를 계속 업데이트
        return x

    # 예측값을 받아서 loss를 내는 역할
    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers): # reversed 잊지 않기!!!!
            dout = layer.backward(dout)

        return dout