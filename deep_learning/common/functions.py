# functions list
import numpy as np

def identity_function(x):
    return x



def sigmoid(x):
    return 1 / (1 + np.exp(-x))