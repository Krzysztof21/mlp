import math as m
import numpy as np
import time
import random as rand

class Neuron:

    def __init__(self, weights, actFunc = 'tanh'):
        self.actFunc = actFunc
        self.weights = np.concatenate((np.array([1]), np.array(weights)))
        self.summ = None
        self.output = None
        self.error = None

    def neuronOutput(self, inp):
        self.inputs = np.concatenate((np.array([1]), np.array(inp)))
        if len(self.inputs) != len(self.weights):
            return "Input shape: " + str(len(self.inputs)) + ", not matching weights size: " + str(len(self.weights))
        else:
            self.summ = np.dot(self.weights, self.inputs)
            if self.actFunc == 'tanh':
                self.output = m.tanh(self.summ)
            elif self.actFunc == 'sigmoid':
                self.output = 1/(1 + m.exp(-self.summ))
        return self.output

class Layer:

    def __init__(self, noNeurons, weights, function = 'tanh', name = '<Name not specified>'):
        self.noNeurons = noNeurons
        self.Neurons = []
        self.function = function
        self.weights = weights
        self.name = name
        self.output = None
        self.summ =None
        for i in range(noNeurons):
            self.Neurons.append(Neuron(self.weights[i], self.function))

    def __repr__(self):
        return "\nName: " + str(self.name) +  ", no. of neurons " + str(self.noNeurons) + ", no. of weights " + str(len(self.weights)) + ", activation function: " + str(self.function)

    def layerOutput(self, input):
        self.output = []
        self.summ = []
        for i in range(len(self.Neurons)):
            self.output.append(self.Neurons[i].neuronOutput(input))
            self.summ.append(self.Neurons[i].summ)
        return self.output


class Network:

    Layers = []

    def __init__(self, inputSize, outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize

    def __addLayer(self, layer):
        self.Layers.append(layer)

    def addArbLayer(self, noNeurons, function = 'tanh'):
        if len(self.Layers) == 0:
            weights = np.random.rand(noNeurons, self.inputSize)
        else:
            weights = np.random.rand(noNeurons, len(self.Layers[len(self.Layers)-1].Neurons))
        name = "FC" + str(len(self.Layers))
        layer = Layer(noNeurons, weights, function, name)
        self.__addLayer(layer)

    def addInputLayer(self, noNeurons, function = 'tanh'):
        weights = np.ones((noNeurons, self.inputSize))
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
            self.output = np.array(self.output)
            return np.array(self.output)

    def costFunction(self, netWeights, input, truth, lmbda):
        start = 0
        for i in self.Layers:
            end = start + i.weights.shape[0]*i.weights.shape[1]
            i.weights = netWeights[start:end].reshape((i.weights.shape[0], i.weights.shape[1]))
            start = end
        input = np.array(input)
        truth = np.array(truth)
        m = len(truth)
        err = 0
        for i in range(m):
            output = np.array(self.forwardPass(input[i]))
            temp1 = np.multiply(truth[i], np.log(output))
            temp2 = np.multiply(1 - truth[i], np.log(1 - output))
            err += np.sum(temp1 + temp2)

        reg = 0
        for layer in self.Layers:
            wgh = np.array(layer.weights)
            reg += np.sum(np.sum(np.power(wgh, 2)))

        return np.sum(err / (-m)) + reg * lmbda / (2 * m)

    def tanhVector(self, x):
        a = []
        for i in x:
            a.append(m.tanh(i))
        return np.array(a)

    def backpropagation(self, netWeights, input, truth, lmbda):
        start = 0
        for i in self.Layers:
            end = start + i.weights.shape[0] * i.weights.shape[1]
            i.weights = netWeights[start:end].reshape((i.weights.shape[0], i.weights.shape[1]))
            start = end
        input = np.array(input)
        truth = np.array(truth)
        mm = len(truth)
        deltas = [list()] * (len(self.Layers) - 1)
        for i in range(mm):
            dHigher = np.array(self.forwardPass(input[i]) - truth[i]).astype(float)
            for j in [3,2,1]:
                b = np.multiply(np.array(self.Layers[j].weights).T, dHigher.T)
                c = (1 - self.tanhVector(self.Layers[j].summ))
                d = np.matmul(b, c.T)
                if i == 0:
                    deltas[j-1] = np.outer(d, np.array(self.Layers[j-1].output))
                    h = 1
                else:
                    deltas[j-1] += np.outer(d, np.array(self.Layers[j-1].output))
                dHigher = d.astype(float)
        print(deltas)
        for i in deltas:
            i /= mm
        flatDeltas = np.empty((0, 1))
        for i in deltas:
            flatDeltas = np.append(flatDeltas, i.flatten())
        return flatDeltas

