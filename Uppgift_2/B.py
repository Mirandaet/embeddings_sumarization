import numpy as np

def sigmoid(x):
    y = 1 / (1 + np.exp(-x)) 
    return y



class Perceptron():
    def __init__(self, architechture):
        weights_values = {}
        bias_values = {}

        for i, layer in enumerate(architechture):
            input_size = layer["input_size"]
            output_size = layer["output_size"]

            weights_values["w" + "_" +
                          str(i)] = np.random.randn(output_size, input_size) * 0.1
            bias_values["b" + "_" +
                        str(i)] = np.random.randn(output_size, 1) * 0.1

        self.weights = weights_values
        self.bias = bias_values

    def forward(self, x):
        for i in range(len(self.weights)):
            x = np.dot(self.weights["w_" + str(i)], x) + self.bias["b_" + str(i)]
            x = sigmoid(x)
        return x
        

architecture = [
    {"input_size": 3, "output_size": 5},  
    {"input_size": 5, "output_size": 2}   
]

parameters = Perceptron(architecture)

x = inputs = np.array([1, 0.2, 0.9])

print(parameters.forward(x))