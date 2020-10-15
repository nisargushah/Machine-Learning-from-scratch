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

def sigmoid_derivative(x):
    return x*(1-x)

def compute_delta(cache,target,layer_no,delta_next, parameters,output_layer = False):
    if output_layer == True:
        delta = (cache - target.reshape(target.shape[0],1))*sigmoid_derivative(cache)
        #print(sigmoid(cache).shape)
    else:
        delta = np.sum(delta_next*parameters['W'+str(layer_no+1)])*sigmoid_derivative(cache)
        #print(delta.shape)
    return delta

def updateParameter(parameters,delta,learning_rate,A_prev, layer_no):
    grads = {}
    delta = np.asarray(delta)
    #print(parameters['W'+str(layer_no)].shape,delta.shape,A_prev.shape )
    #print()
    #grads['W'+str(layer_no)] = parameters['W'+str(layer_no)]  - learning_rate*(delta*A_prev)
    #print("hello")
    print(parameters['W'+str(layer_no)]  - learning_rate*(delta*A_prev))
    grads['b'+str(layer_no)] = parameters['b'+str(layer_no)] - learning_rate*delta
    return grads

def initial_weight(units):
    parameters = {}
    L = len(units)
    print("L = ", L)
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.uniform(-0.5,0.5,size=(units[l],units[l-1]))
        #print(parameters['W1'])
        parameters['b' + str(l)] = np.zeros((units[l], 1))
        assert(parameters['W' + str(l)].shape == (units[l], units[l - 1]))
        assert(parameters['b' + str(l)].shape == (units[l], 1))
    return parameters

def sigmoid(x):
    return (1 / (1+np.exp(-x))).astype(np.float128)

def feed_forward(A_prev, W, b):
    x = A_prev.shape[0]
    A_prev.reshape(x,1)
    Z = np.dot(W, A_prev.reshape(x,1)) + b
    """print(W.transpose().shape)
    #print(Z)
    print(Z.shape, A_prev.reshape(x,1).shape[1])"""
    assert(Z.shape == (W.shape[0], A_prev.reshape(x,1).shape[1]))
    cache = (A_prev, W, b)
    return Z ,cache, sigmoid(Z)

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
    X_train = (X_train/X_train.max()).astype(np.float64)
    X_test = (X_test/X_test.max()).astype(np.float64)
    vector_train = one_hot_encoder(y_train.astype(np.int))
    vector_test = one_hot_encoder(y_test.astype(np.int))
    #print(vector_train[0].shape)

    units = []

    #print(units)
    units = [units_per_layer for i in range(1,layers-1)]

    units.insert(0,X_train.shape[1])
    numOfClasses = np.max(y_train) - np.min(y_test) +1
    units.append(int(numOfClasses))
    #print(units)
    parameters = initial_weight(units)
    print(len(units)-1)

    delta_next = 0
    #print(X_test.shape)
    for k in range(2):
        cache= []
        delta = []
        loss = 0
        A_list = []
        Z_master = []
        delta_list = []
        delta_list.append(0)
        print(parameters['W1'][0])
        for j in range(len(X_train)):
            caches_per_layer = []
            A = X_train[j]
            Z_list = []
            for i in range(1,len(units)):
                #print(i)
                A_prev = A
                #print(A_prev.shape, parameters['W' + str(i)].shape )
                Z,cache, A = feed_forward(A_prev,parameters['W' + str(i)],parameters['b' + str(i)])
                caches_per_layer.append(cache)
                Z_list.append(Z)
                A_list.append(A)
                #print(A.shape, Z.shape)
            Z_master.append(Z_list)
            loss += compute_loss(A,vector_train[j])

        #print(caches_per_layer[-1][0].shape)
        #print(len(Z_master[-1][-1]))
        delta_dict = {}
        for j in range(len(X_train)):
            learning_rate = 0.98
            for i in reversed(range(1,len(units))):
                if i == len(units)-1:
                    delta_layer = compute_delta(sigmoid(Z_master[j][i-1]),vector_train[j],i,delta_next, parameters,output_layer = True)
                    delta_dict['delta'+ str(i)] = delta_layer
                    #print( Z_list[i+j-2])
                    #parameters = updateParameter(parameters,delta_layer, learning_rate,sigmoid(Z_master[j][i-1]),i)
                else:
                    delta_layer = compute_delta(sigmoid(Z_master[j][i-1]),vector_train[j],i,delta_next, parameters,output_layer = False)
                    #delta_list.insert(i,delta_layer)
                    delta_dict['delta'+ str(i)] = delta_layer
                    #parameters =  updateParameter(parameters,delta_layer, learning_rate,sigmoid(Z_master[j][i-1]),i)

        learning_rate *= 0.98
        """print("round = ", end='')
        print(k,loss)"""
    #print(parameters['W1'])






if __name__ == '__main__':
    neural_network(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
