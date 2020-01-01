from layer import Layer
import numpy as np
class FCLayer(Layer):
    def __init__(self,input_size,output_size):
        self.weights = np.random.rand(input_size,output_size)
        self.bias = np.random.rand(1,output_size)

    def forwardProp(self,input_data):
        self.input = input_data
        self.output = np.dot(self.input,self.weights) + self.bias
        return self.output

    def backwardProp(self,output_error,learing_rate):
        weight_error = np.dot(self.input.T,output_error)
        input_error = np.dot(output_error,self.weights.T)
        self.weights = self.weights - weight_error*learing_rate
        self.bias = self.bias - output_error * learing_rate
        return  input_error
