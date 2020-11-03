import sys
import numpy as np
from collections import Counter
from statistics import stdev

def normalise(data, test_data):
    data = np.asarray(data)
    #test_data = np.asarray(test_data, dtype=np.float32)

    stats = [[np.mean(data[:,i]), np.std(data[:,i])] for i in range(0,data.shape[1]-1)]
    #print(stats)
    for x in range(0, data.shape[0]):
        for y in range(0,data.shape[1]-1):
            data[x][y] = (data[x][y] - stats[y][0])/stats[y][1]
    for x in range(0, test_data.shape[0]):
        for y in range(0,test_data.shape[1]-1):
            test_data[x][y] = (test_data[x][y] - stats[y][0])/stats[y][1]
    return data, test_data

def nearest_neighbor(train_data, test_data, k):
    object_id = 1
    accuracy = []
    for predict in test_data:
        distance = []
        for data in train_data:
            temp_dist =np.sqrt(np.sum(np.square(np.array(data[:-1]) - np.array(predict[:-1]))))
            distance.append((temp_dist,data[-1]))
        votes = [i[1] for i in sorted(distance)[:k]]
        predicted_class = Counter(votes).most_common(1)[0][0]
        votes = np.asarray(votes)
        print(object_id, votes)
        if len(np.unique(votes)) == k:
            if predict[-1] in votes:
                accuracy.append(1/k)
            else:
                accuracy.append(0)
        else:
            accuracy.append(1) if predicted_class == predict[-1] else accuracy.append(0)
        #print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f'% (object_id, predicted_class, predict[-1], accuracy[-1]));
        object_id += 1
    print('classification accuracy=%6.2f'%(sum(accuracy)/len(accuracy)*100))

if __name__ == '__main__':
    train_data,test_data  = normalise(np.loadtxt(sys.argv[1]),np.loadtxt(sys.argv[2]) )
    data = np.loadtxt(sys.argv[1])
    k = int(sys.argv[3])
    print(len(train_data[0])-1)
    print(train_data.shape)
    nearest_neighbor(train_data, test_data, k)
