
class Layer:
    def __init__(self):
        self.input = None 
        self.output = None

    def forwardProp(self,input):
        raise NotImplementedError
    
    def backwardProp(self,output,learing_rate):
        raise NotImplementedError