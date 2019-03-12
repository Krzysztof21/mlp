import csv
import numpy as np

data = np.genfromtxt('IrisDataTrain.csv', delimiter='",',dtype=str)
for i in range(len(data)):
     for j in range(len(data[1])):
         data[i,j] = data[i,j].replace('\"','').replace(',','.')
inp = data[:,0:4].astype(float)

maxima = np.amax(inp, axis=0)
for i in range(len(inp)):
    inp[i] = np.divide(inp[i],maxima)


print(inp) 

