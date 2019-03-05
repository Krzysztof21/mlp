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

class Layer:

    def __init__(self, inp, n):




#####################################################################

inp = [1,2,3,4]
initweights = [0.5] * 4

n = Neuron(inp, initweights)

print(n.S)
print(n.output())

