import numpy as np

class Smooth():
    
    @staticmethod
    def plus(x,alpha):
        return x + (1/alpha)*np.log(1+np.exp(-alpha*x))
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1+np.exp(-x))
    
    

