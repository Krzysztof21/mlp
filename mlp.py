import math as m
import numpy as np
import time
import random as rand

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

    def __init__(self, noNeurons, weights, function = 'tanh', name = '<Name not specified>'):
        self.noNeurons = noNeurons
        self.Neurons = []
        self.function = function
        self.weights = weights
        self.name = name
        for i in range(noNeurons):
            self.Neurons.append(Neuron(self.weights, self.function))

    def __repr__(self):
        return "\nName: " + str(self.name) +  ", no. of neurons " + str(self.noNeurons) + ", no. of weights " + str(len(self.weights)) + ", activation function: " + str(self.function)

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

    def __addLayer(self, layer):
        self.Layers.append(layer)

    def addArbLayer(self, noNeurons, function = 'tanh'):
        if len(self.Layers) == 0:
            weights = [rand.random()] * self.inputSize
        else:
            weights = [rand.random()] * len(self.Layers[len(self.Layers)-1].Neurons)
        name = "FC" + str(len(self.Layers))
        layer = Layer(noNeurons, weights, function, name)
        self.__addLayer(layer)

    def addInputLayer(self, noNeurons, function = 'tanh'):
        weights = [rand.random()] * self.inputSize
        name = "Input"
        layer = Layer(noNeurons, weights, function, name)
        if len(self.Layers) == 0:
            self.Layers.append(layer)
        else:
            self.Layers.insert(0, layer)
            self.Layers[1].weights = [rand.random()] * noNeurons

    def forwardPass(self, input):
        if len(input) != self.inputSize:
            return "Input size does not match expected size"
        else:
            self.output = input
            for i in range(len(self.Layers)):
                self.output = self.Layers[i].layerOutput(self.output)
            return self.output
