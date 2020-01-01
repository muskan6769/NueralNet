from layer import Layer
import numpy as np
class ActivationLayer(Layer):
    def __init__(self,Activation,Activation_prime):
        self.Activation = Activation
        self.Activation_prime = Activation_prime

    def forwardProp(self,input_data):
        self.input = input_data
        self.output = self.Activation(input_data)
        return self.output

    def backwardProp(self,output_error,learing_rate):   
        input_error = output_error*self.Activation_prime(self.input)
        return  input_error
