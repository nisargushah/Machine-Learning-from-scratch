"""
Nisarg Shah

1001553132

"""

import sys
import math
import numpy as np
from statistics import stdev as stdev

def gaussian(x, mean=0.0, sigma=1.0):
    temp = float((x-mean)/sigma)
    e_factor = np.exp(-(np.power(temp,2) / 2))
    deno = sigma*(np.sqrt(2*np.pi))
    return e_factor / deno


def naive_bayes(train_file, test_file):
    #print(train_file, test_file)
    try:
        import pandas as pd
        train_data = pd.read_csv(train_file, header = None)
        test_data = pd.read_csv(test_file, header=None)
        X_test  = train_data.iloc[:,:-1]
        X_test = X_test.astype(np.float)
        y_test  = train_data.iloc[:,-1]
        y_test = y_test.astype(np.int)
        X_train = test_data.iloc[:,-1]
        X_train = X_train.astype(np.float)
        y_train = test_data.iloc[:,-1]
        y_train = y_train.astype(np.int)
        #print("From Pandas")
    except:
        train_data = np.genfromtxt(train_file)
        test_data = np.genfromtxt(test_file)
        X_test = test_data[:, :-1]
        X_train = train_data[:, :-1]
        y_test = test_data[:, -1]
        y_train = train_data[:, -1]
        X_test = X_test.astype(np.float)
        y_test = y_test.astype(np.int)
        X_train = X_train.astype(np.float)
        y_train = y_train.astype(np.int)
    #Seperating Training examples and labels.
    #print(X_train)
    class_means = []
    class_std = []
    #print(X_train[0][1])
    indexes = []
    y_train = np.asarray(y_train)
    for i in range(1,11):
        x = np.where(y_train == i)
        #print(x)
        x = np.asarray(x)
        temp = []
        #print(x[0])
        for j in range(0, x.shape[1]):
            temp.append(X_train[x[0][j],:])
        #print(temp)
        temp = np.asarray(temp)
        for j in range(0,8):
            temp2 = temp[:,j]
            temp2 = np.asarray(temp2)
            mean = temp2.mean()
            #std  = temp2.std()
            std = stdev(temp2)
            if std<=0.01:
                std = 0.01
            class_means.append(mean)
            class_std.append(std)
            #print("mean %.2f std %.2f" %(mean, std))
    a = 0
    for i in range(1,11):
        for j in range(1,9):
             print("Class %d, attribute %d, mean = %.2f, std = %.2f" % (i, j, class_means[a], class_std[a]))
             a+=1
    #Finding the prior Probability for each class.
    num_classes = len(np.unique(y_train))
    #print(num_classes)
    #print(y_train.min())
    min_class = y_train.min()
    prior_prob = []
    training_examples = (X_train.shape[0])
    for i in range(min_class,num_classes+1):
        ind = np.where(y_train == i)
        ind = np.asarray(ind)
        p_C = ind.shape[1]/training_examples
        prior_prob.append(p_C)
        #print(" Class ",i, "Prior Probability " ,p_C)
    prior_prob = np.asarray(prior_prob)
    final_prob = []
    class_final = []
    a = 0
    q=0
    p_x_given_C = 1
    ###Classification Stage:
    for i in range(0,len(y_test)):
        temp_prob = []
        for j in range(0, num_classes):
            for k in range(0,X_train.shape[1]):
                attr = X_test[i][k]
                p_x_given_C *= gaussian(attr, class_means[a], class_std[a])
                #print(p_x_given_C)
                a+=1
            prob = p_x_given_C*prior_prob[j]
            #print(p_x_given_C)
            #print(prob)
            temp_prob.append(prob)
            p_x_given_C = 1
        a=0
        q=0
    #print(X_train.shape[1])
        temp_prob[:] = [x/np.sum(temp_prob) for x in temp_prob]
        final_prob.append(max(temp_prob))
        #print(final_prob)
        class_final.append(temp_prob.index(max(temp_prob)))
        temp_prob.clear()
    acc = []
    for i in range(0,len(class_final)):
        if class_final[i]+1 == y_test[i]:
            acc.append(1)
        else:
            acc.append(0)
        print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n"%(i+1, class_final[i]+1, final_prob[i], y_test[i], acc[i]))
        P=1
    acc = np.asarray(acc)
    print("Accuracy : %.4f"%(np.sum(acc)/len(acc)) )

if __name__ == "__main__":
    naive_bayes(sys.argv[1],sys.argv[2])
