import numpy as np
import sys


'''
This function will be used to return sigmoid


'''
def sigmoid(x):
    return 1 / (1+np.exp(-x))

def initial_weight(rows, columns):
    w = np.random.uniform(-0.5,0.5,(rows, columns))
    b = np.zeros((columns,1))
    return w,b



def one_hot_encoder(a):
    if a.min() == 0:
        b = np.zeros((a.size, a.max()+1))
        b[np.arange(a.size),a] = 1
        return b
    else:
        b = np.zeros((a.size, a.max()))
        b[np.arange(a.size),a-1] = 1
        return b

def feed_forward(W,X,b):

    Z = np.dot(W,X) + b
    A = sigmoid(Z)
    return A

def cost(t,Z):
    assert t.shape == Z.shape
    return (1 / 2)*np.sum(np.square(t-Z))



def neural_network(train_file, test_file, layers, units_per_layer, rounds):
    ##First step is to read in the files.

    train_data = np.loadtxt(train_file)
    #print(train_data)
    test_data = np.loadtxt(test_file)
    #print(test_data)
    X_train = train_data[:,:-1]
    y_train = train_data[:,-1].astype(int)
    X_test = test_data[:,:-1]
    y_test = test_data[:,-1]


    max1 = max(X_train.min(), X_train.max(), key=abs)
    X_train = X_train/max1
    X_test = X_test/max1

    ##First we will check if we have correct inputs:
    if int(layers) < 2:
        print("Please enter correct number of input layers:")
        return 0
    one_hot_y = one_hot_encoder(y_train)
    x = X_train[0]
    x = x.reshape(X_train.shape[1],1)
    #print(x.shape)

    w,b = initial_weight(X_train.shape[1], 1)
    z = feed_forward(w,x.transpose(),b)
    for i in range(1,layers):

        if i == layers-1:
            w,b = initial_weight(len(np.unique(y_train)),1)
        else:

            w,b = initial_weight(units_per_layer,z.shape[1])
        z = feed_forward(w,z.transpose(),b)
        print(w.shape,z.shape)




















if __name__ == '__main__':
    x = neural_network(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]))




# neural_network(<training_file>, <test_file>, <layers>, <units_per_layer>, <rounds>)
