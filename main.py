import math as m
import mlp
import matplotlib.pyplot as plt
import scipy.stats as sci
import numpy as np



inp = [1,2,3,4]
initweights4 = [0.5] * 4
initweights8 = [0.5] * 8

n = mlp.Neuron(initweights4)

lay1 = mlp.Layer(8, initweights8)

lay2 = mlp.Layer(4, initweights8)


net = mlp.Network(8,4)


net.addArbLayer(8)
net.addArbLayer(4)
net.addArbLayer(4)
net.addInputLayer(8)

inp2 = [1,2,3,4,3,12,4,5]
print(net.Layers)

print(net.forwardPass(inp2))

