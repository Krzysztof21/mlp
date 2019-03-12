import math as m

class Neuron:

    S = 0

    def __init__(self, inp, wei):
        self.inputs = inp
        self.weights = wei
        for i in range(len(self.inputs)):
            self.S = self.S + self.weights[i] * self.inputs[i]


    def output(self):
        return m.tanh(self.S)