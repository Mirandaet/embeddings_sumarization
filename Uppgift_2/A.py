import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()

    def forward(self, inputs):
        return sigmoid(np.dot(self.weights, inputs) + self.bias)

class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

class NeuralNetwork:
    def __init__(self, architecture):
        self.layers = [Layer(neurons, input_size) for neurons, input_size in architecture]

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

if __name__ == "__main__":
    architecture = [(20, 3), (10, 20), (1, 10)]
    network = NeuralNetwork(architecture)
    input_data = np.array([0.5, -0.2, 0.1])
    prediction = network.predict(input_data)
    print(prediction)