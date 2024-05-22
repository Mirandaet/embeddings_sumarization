```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.normal(0, 0.01, num_inputs)
        self.bias = np.random.normal(0, 0.01)

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return sigmoid(z)

class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

class NeuralNetwork:
    def __init__(self, architecture):
        self.layers = [Layer(neurons, inputs) for neurons, inputs in architecture]

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

# Example of creating a network and making a prediction
if __name__ == "__main__":
    architecture = [(20, 3), (10, 20), (1, 10)]
    network = NeuralNetwork(architecture)
    input_data = np.array([0.5, -0.2, 0.1])
    prediction = network.predict(input_data)
    print(prediction)
```