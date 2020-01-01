
class Network:
    def __init__(self):
        self.layers=[]
        self.loss = None
        self.loss_prime = None

    def add(self,layer):
        self.layers.append(layer)

    def use(self,meanSqError,meanSqError_prime):
       self.loss = meanSqError
       self.loss_prime = meanSqError_prime


    def fit(self, x_train, y_train, epochs, learning_rate):
        for i in range(0,epochs):
            input_data = x_train
            
            for k in range(0,len(input_data)):           
                # print(input_data.iloc[k])
                output=input_data[k]
                for j in self.layers:
                    output = j.forwardProp(output)
                error = self.loss(output,y_train[k])

                output_error = self.loss_prime(output,y_train[k])

                for j in reversed(self.layers):
                    input_error = j.backwardProp(output_error,learning_rate)
                    output_error = input_error

    def predict(self,x_test):
        samples = len(x_test)
        result = []
        for i in range(samples):       
            output = x_test[i]
            for j in self.layers:
                    output = j.forwardProp(output)
                    
            result.append(output)
        return result
     

