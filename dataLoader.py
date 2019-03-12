import numpy as np

def encode(table):
    t = table[:,4]
    output = np.zeros((len(t),3),dtype=float)

    for i in range(len(t)):
        if t[i] == 'Iris-setosa':
            output[i] = [1,0,0]
        elif t[i] == 'Iris-versicolor':
            output[i] = [0,1,0]
        elif t[i] == 'Iris-virginica':
            output[i] = [0,0,1]
    return output

data = np.genfromtxt('IrisDataTrain.csv', delimiter='",',dtype=str)

for i in range(len(data)):
     for j in range(len(data[1])):
         data[i,j] = data[i,j].replace('\"','').replace(',','.')

inp = data[:,0:4].astype(float)

maxima = np.amax(inp, axis=0)
for i in range(len(inp)):
    inp[i] = np.divide(inp[i],maxima)

output = encode(data)


