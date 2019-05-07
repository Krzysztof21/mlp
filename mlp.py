import math as m
import numpy as np
import time

class Neuron:

    def __init__(self, weights, actFunc = 'tanh'):
        self.actFunc = actFunc
        self.weights = np.concatenate((np.array([1]), np.array(weights)))

    def neuronOutput(self, inp):
        self.inputs = np.concatenate((np.array([1]), np.array(inp)))
        if len(self.inputs) != len(self.weights):
            return "Input shape: " + str(len(self.inputs)) + ", not matching weights size: " + str(len(self.weights))
        else:
            self.sum = np.sum(self.weights * self.inputs)
            if self.actFunc == 'tanh':
                return m.tanh(self.sum)
            elif self.actFunc == 'sigmoid':
                return 1/(1 + m.exp(-self.sum))

class Layer:

    def __init__(self, noNeurons, weights, function = 'tanh'):
        self.noNeurons = noNeurons
        self.Neurons = []
        self.function = function
        self.weights = weights
        for i in range(noNeurons):
            self.Neurons.append(Neuron(self.weights, self.function))

    def __repr__(self):
        return "\nNo. of neurons " + str(self.noNeurons) + ", activation function: " + str(self.function)

    def layerOutput(self, input):
        self.input = input
        self.out = []
        for i in range(len(self.Neurons)):
            self.out.append(self.Neurons[i].neuronOutput(self.input))
        return self.out


class Network:

    Layers = []

    def __init__(self, inputSize, outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize

    def addArbLayer(self, noNeurons, function = 'tanh'):
        weights = [0.5] * len(self.Layers[len(self.Layers)-1].Neurons)
        layer = Layer(noNeurons, weights, function)
        self.addLayer(layer)

    def addLayer(self, layer):
        self.Layers.append(layer)

    def addInputLayer(self, layer):
        if len(self.Layers) == 0:
            self.Layers.append(layer)
        else:
            self.Layers.insert(0, layer)

    def getNetOutput(self, input):
        return self.Layers[len(self.Layers)-1].layerOutput(input)