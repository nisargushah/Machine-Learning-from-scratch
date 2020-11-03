from sklearn.neighbors import KNeighborsClassifier
import sys
import numpy as np
from collections import Counter
from statistics import stdev

def normalise(data, test_data):
    data = np.asarray(data)
    #test_data = np.asarray(test_data, dtype=np.float32)

    stats = [[np.mean(data[:,i]), np.std(data[:,i])] for i in range(0,len(data[0])-1)]
    #print(stats)
    for x in range(0, data.shape[0]):
        for y in range(0,data.shape[1]-1):
            data[x][y] = (data[x][y] - stats[y][0])/stats[y][1]
    for x in range(0, test_data.shape[0]):
        for y in range(0,test_data.shape[1]-1):
            test_data[x][y] = (test_data[x][y] - stats[y][0])/stats[y][1]
    return data, test_data


train_data,test_data  = normalise(np.loadtxt(sys.argv[1]),np.loadtxt(sys.argv[2]) )
neigh = KNeighborsClassifier(n_neighbors=int(sys.argv[3]))
neigh.fit(train_data[:,:-1], train_data[:,-1])
predict = neigh.predict(test_data[:,:-1])
accuracy = []
for i in range(len(predict)):
    if predict[i] == test_data[i,-1]:
        accuracy.append(1)
    else:
        accuracy.append(0)

print("Final accuracy : %6.4f"%( 100*sum(accuracy)/len(accuracy)))
