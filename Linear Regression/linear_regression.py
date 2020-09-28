import numpy as np
import os
import sys

def linear_regression(train_file, test_file, degree, lam):
    #First step is to read in the data
    # We will use numpy for this, however we can also use pandas.
    train_data = np.loadtxt(train_file, dtype=np.float32)

    # Quick debug check
    #print(train_data.shape)

    test_data = np.loadtxt(test_file, dtype= np.float32)
    # Quick debug check
    #print(test_data.shape)

    # Main task here will be using the formula given to us on slide 61
    # First lets seperate our classes.
    y_train = train_data[:,-1]
    y_test = test_data[:,-1]
    X_train = train_data[:,:-1]
    X_test = test_data[:,:-1]

    # Here out target t will be the y_train.

    # Training Stage:
    # lets get phi first
    phi = np.ones([X_train.shape[0], (X_train.shape[1])*degree + 1])

    for i in range(X_train.shape[0]) :
        x=0
        z=1
        while z<phi.shape[1] :
            for j in range(1,degree+1) :
                phi[i][z] = np.power(X_train[i][x],j)
                z=z+1
            x=x+1
    phi_transpose = np.transpose(phi)
    mult = np.matmul(phi_transpose, phi)
    term = np.linalg.pinv(lam*np.identity(len(mult))+ mult)
    w = np.matmul(np.matmul(term, phi_transpose), y_train)
    #print(w)

    #Testing Stage

    phi_test = np.ones([X_test.shape[0], (X_test.shape[1])*degree + 1])
    for i in range(phi_test.shape[0]) :
        q=0
        a=1
        while a<phi_test.shape[1] :
            for j in range(1,degree+1) :
                phi_test[i][a] = np.power(X_test[i][q],j)
                a=a+1
            q=q+1
    #print(phi_test.shape)

    for index in range(len(w)):
        print("w%d=%.4f"%(index,w[index]))

    for i in range(len(y_test)):
        result = np.dot(w,phi_test[i])
        print("ID = " + "%5d"%int(i+1) + ", output = " + "%14.4f" %result + ", target_value=" + "%10.4f" %y_test[i] + ", squared error = " + "%0.4f" %(np.square(y_test[i]-result)))

if __name__== '__main__':
    linear_regression(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
