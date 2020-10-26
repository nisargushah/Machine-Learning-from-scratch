import sys
import numpy as np
from utils import fetch_data
from utils import distribution
from utils import information_gain


def DTL_TopLevel(examples,classes, option, threshold):
    """
    This function will be served as a top level Decision Tree and
    will just return our tree as displayed in the Psuedo Code

    """

    attr = examples.shape[1]
    default = distribution(classes)
    return DTL(examples, classes, attr, default,  option, threshold)

def DTL(examples,classes, attr, default, option, threshold):
    """
    """
    #First lets check for base case
    print(examples.shape)
    if len(examples) == 0:
         return default
    elif len(np.unique(classes)) == 1:
        return classes
    else:
        best_attribute, best_threshold = choose_attr(examples, attr, option, classes)
        target_attr = examples[:,  best_attribute]
        left = examples[examples[:,best_attribute] < best_threshold]
        right = examples[examples[:,best_attribute] >= best_threshold]
        left = np.asarray(left)
        print(left.shape)
        right = np.asarray(right)
        left_DLT = DTL(left, classes, attr,distribution(classes),option, threshold)
        right_DLT = DTL(right, classes, attr,distribution(classes),option, threshold)

        tree ={
                "left": left_DLT,
                "right": right_DLT
        }
        return tree

def choose_attr(examples, attr, option, classes):

    if option == 'optimized':
        max_gain = best_attribute = best_threshold = -1
        #print(examples.shape)
        for i in range(attr):
            attr_value = np.array(examples)[:,i]
            #print(attr_value.shape)
            L = np.min(attr_value)
            M = np.max(attr_value)
            #print(i)
            for K in range(1,51):
                threshold = L + K*(M-L)/51
                gain = information_gain(examples,classes,i, threshold)
                #print(gain)
                if gain > max_gain:
                    best_attribute = i
                    best_threshold = threshold
        return best_attribute, best_threshold


    elif option == 'randomized':
        pass

    elif option  =='forest3':
        pass

    elif option == 'forest15':
        pass

    else:
        print("Please enter the correct option")
        sys.exit(0)


if __name__ == '__main__':
    X_train, y_train = fetch_data(sys.argv[1])
    X_test , y_test  = fetch_data(sys.argv[2])
    mode = sys.argv[3]
    prun = sys.argv[4]
    print(DTL_TopLevel(X_train,y_train, mode, prun))
