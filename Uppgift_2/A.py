```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.normal(0, 1.0, num_inputs)
        self.bias = np.random.normal()

    def forward(self, inputs):
        return sigmoid(np.dot(self.weights, inputs) + self.bias)

class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

class NeuralNetwork:
    def __init__(self, architecture):
        self.layers = [Layer(neurons, (architecture[i-1][0] if i > 0 else inputs)) 
                       for i, (neurons, inputs) in enumerate(architecture)]

    def predict(self, inputs):
        return np.array([np.clip(layer.forward(inputs), -1, 1) for layer in self.layers])[-1]

if __name__ == "__main__":
    architecture = [(20, 3), (10, 20), (1, 10)]
    network = NeuralNetwork(architecture)
    input_data = np.array([0.5, -0.2, 0.1])
    prediction = network.predict(input_data)
    print(prediction)
```