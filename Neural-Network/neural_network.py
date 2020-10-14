import sys
import os
import numpy as np

def extract_data(file):
    data = np.loadtxt(file)
    class_data = data[:,-1]
    attr_data = data[:,:-1]
    return attr_data,class_data

def one_hot_encoder(data):
    if data.min() == 0:
        vector = np.zeros((data.size, data.max()+1))
        vector[np.arange(data.size),data] = 1
        return vector
    else:
        vector = np.zeros((data.size, data.max()))
        vector[np.arange(data.size),data-1] = 1
        return vector


def initial_weight(units):
    parameters = {}
    L = len(units)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.uniform(-0.5,0.5,size=(units[l],units[l-1]))
        parameters['b' + str(l)] = np.zeros((units[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

def sigmoid(x):
    return (1 / (1+np.exp(-x)))

def feed_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    cache = (A_prev, W, b)
    return Z, cache, sigmoid(Z)




def neural_network(train_file, test_file, layers, units_per_layer, rounds):

    ##Read in the files
    X_train, y_train = extract_data(train_file)
    X_test, y_test = extract_data(test_file)
    #print(X_train)

    ##Converting our class_data to one hot encoder
    """print(y_test)
    print(one_hot_encoder(y_test.astype(np.int))[0])"""

    ##Normalizign the data on X_train and X_test
    X_train = X_train/X_train.max()
    X_test = X_test/X_test.max()

    units = []

    #print(units)
    units = [units_per_layer for i in range(1,layers+1)]

    units.insert(0,X_train.shape[0])
    numOfClasses = np.max(y_train) - np.min(y_test) +1
    units.append(int(numOfClasses))
    print(units)
    #print(X_test.shape)




if __name__ == '__main__':
    neural_network(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
