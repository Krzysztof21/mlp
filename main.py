import math as m
import mlp
import matplotlib.pyplot as plt
import scipy.stats as sci
import numpy as np


#------------------------------------------------------------

inp = [1,2,3,4]
initweights = [0.5] * 4

n = mlp.Neuron(initweights)

lay1 = mlp.Layer(4,initweights)
lay1.layerOutput(inp)
print(lay1.out)

lay2 = mlp.Layer(8, initweights)
lay2.layerOutput(inp)
print(lay2.out)

net = mlp.Network(4,4)
net.addLayer(lay1)
net.addLayer(lay2)
net.addInputLayer(lay2)
weightsSix = [0.5]*8
net.addArbLayer(8)

inp2 = [1,2,3,4,3,12,4,5]
print(net.Layers)
print(net.getNetOutput(inp2))

#import dataLoader

#plt.plot(inp, initweights)
#plt.show()

