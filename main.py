import math as m
import mlp
import numpy as np


net = mlp.Network(8,4)

net.addInputLayer(8)
net.addArbLayer(8)
net.addArbLayer(4)
net.addArbLayer(4)

X = np.array([[3, 2, 7, 1, 3, 2, 3, 5], [1, 2, 3, 4, 3, 12, 4, 5]])
y = np.array([[0, 0, 1, 0], [1, 0, 0, 0]])

print(net.Layers)

lmbda = 1 

netWeights = np.empty((0,1))
for i in net.Layers:
    netWeights = np.append(netWeights, i.weights.flatten())


# calculating output of the network with a given input
print(net.forwardPass(X[1]))

# calculating cost of using a given set of weights
print(net.costFunction(netWeights, X, y, 1))

# calculation of gradient for each of the weights
print(net.backpropagation(netWeights, X, y, 1))

