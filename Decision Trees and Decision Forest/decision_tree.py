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

    if len(examples) == 0:
         return default
    elif len(np.unique(classes)) == 1:
        return classes
    else:
        best_attribute, best_threshold = choose_attr(examples, attr, option, classes)
        target_attr = examples[:,  best_attribute]
        left = [target_attr < threshold]
        right = [target_attr >= threshold]

        left_DLT = DLT(left, classes, attr,distribution(classes),option, threshold)
        right_DLT = DLT(right, classes, attr,distribution(classes),option, threshold)

        tree ={
                "left": left_DLT,
                "right": right_DLT
        }
        return tree

def choose_attr(examples, attr, option, classes):

    if option == 'optimized':
        max_gain = best_attribute = best_threshold = -1
        for attr_ in range(attr):
            attr_value = np.array(examples[:,attr_])
            L = np.min(attr_value)
            M = np.max(attr_value)
            for K in range(1,51):
                threshold = L + K*(M-L)/51
                gain = information_gain(examples,classes, attr_, threshold)
                print(gain)
                if gain > max_gain:
                    best_attribute = attr_
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
