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
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.uniform(-0.5,0.5,size=(units[l],units[l-1]))
        parameters['b' + str(l)] = np.zeros((units[l], 1))
        assert(parameters['W' + str(l)].shape == (units[l], units[l - 1]))
        assert(parameters['b' + str(l)].shape == (units[l], 1))
    return parameters

def sigmoid(x):
    return (1 / (1+np.exp(-x)))

def feed_forward(A_prev, W, b):
    x = A_prev.shape[0]
    A_prev.reshape(x,1)
    Z = np.dot(W, A_prev.reshape(x,1)) + b
    """print(W.transpose().shape)
    #print(Z)
    print(Z.shape, A_prev.reshape(x,1).shape[1])"""
    assert(Z.shape == (W.shape[0], A_prev.reshape(x,1).shape[1]))
    cache = (A_prev, W, b)
    return (Z, cache), sigmoid(Z)

def compute_loss(computed, original):
    loss = 0
    for i in range(computed.shape[0]):
        loss += np.square(computed[i]-original[i])

    #print(loss)
    return loss


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
    vector_train = one_hot_encoder(y_train.astype(np.int))
    vector_test = one_hot_encoder(y_test.astype(np.int))
    print(vector_train[0].shape)

    units = []

    #print(units)
    units = [units_per_layer for i in range(1,layers+1)]

    units.insert(0,X_train.shape[1])
    numOfClasses = np.max(y_train) - np.min(y_test) +1
    units.append(int(numOfClasses))
    #print(units)
    parameters = initial_weight(units)

    #print(X_test.shape)
    cache= []
    loss = 0
    for j in range(len(X_train)):
        caches_per_layer = []
        A = X_train[j]
        for i in range(1,len(units)):
            #print(i)
            A_prev = A
            #print(A_prev.shape, parameters['W' + str(i)].shape )
            cache, A = feed_forward(A_prev,parameters['W' + str(i)],parameters['b' + str(i)])
            caches_per_layer.append(cache)
            #print(j)
        #cache.append(caches_per_layer)

        print(A.shape, vector_train[j].shape)
        loss += compute_loss(A,vector_train[j])
    #print(caches_per_layer)

    #print(X_train[0].shape)
    #print(loss)



if __name__ == '__main__':
    neural_network(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
