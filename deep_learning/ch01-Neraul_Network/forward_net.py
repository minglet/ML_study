import numpy as np

# sigmoid 구현
class Sigmoid():
    '''Sigmoid Layer class
    
    Sigmoid layer에는 학습하는 params가 따로 없으므로 
    인스턴스 변수인 params는 빈 리스트로 초기화
    
    '''
    def __init__(self):
        self.params = []
    
    def forward(self, x):
        """순전파(forward propagation) 메서드
        Args:
            x(ndarray): 입력으로 들어오는 값
        Returns:
            Sigmoid 활성화 값
        """
        return 1 / (1+ np.exp(-x))

# Affine 완전 연결 계층 구현   
class Affine():
    '''FC layer
    numpy array 타입으로 구성
    '''
    def __init__(self, W, b):
        """
        Args: 
            W(ndarray): 가중치(weight)
            b(ndarray): 편향(bias)
        """
        self.params = [W, b]

    def forward(self, x):
        """순전파(forward propagation) 메서드
        Args:
            x(ndarray): 입력으로 들어오는 값
        Returns:
            out(ndarray): Wx + b
        """
        W, b = self.params
        out = np.matmul(x, W) + b
        return out
    
class TwoLayerNet():
    '''
    input - Affine - Sigmoid - Affine - out
    '''
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 초기화
        # input -> hidden
        W1 = np.random.randn(I, H) 
        b1 = np.random.random(H)
        # hidden -> output
        W2 = np.random.random(H, O)
        b2 = np.random.randn(O)

        # 계층 생성
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 모든 가중치를 리스트에 모은다.
        self.params = []
        for layer in self.layers:
            self.params += layer.params # 리스트로 만들었는데 append하지 않고 더해도 되는것인가?
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
if __name__ == '__main__':
    x = np.random.randn(10, 2)
    model = TwoLayerNet(2, 4, 3)
    s = model.predict(x)
 
    print(f'{s!r}') # !r 이 뜻하는게 뭐지?
