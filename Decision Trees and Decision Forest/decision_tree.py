"""

Nisarg Shah


"""


import sys
import numpy as np
import os
import math
import random
from collections import Counter


class tree(object):
    """docstring for tree."""

    def __init__(self, best_attribute, best_threshold):
        self.best_attribute = best_attribute
        self.best_threshold = best_threshold
        self.left_child  = None
        self.right_child = None
        self.gain = 0
        self.distribution = None

def fetch_data(filename):
    """
    In this function we will be taking in the filename(with relative location) and then
    sperate them to data points or attributes and class labels
    input: filename
    output : X - attributes (with data points)
             y - class labels
    """
    data = np.loadtxt(filename, dtype=np.float32)
    return data

def distribution(examples):
    """
    This funtion helps us to find the distribution of class or the probability of
    each class by finding number if times it occurs in a given set of examples.

    Since we have to pass the final y_train of every examples here, I use theraw data to
    find it. So I can simply find the last column and do operations on them

    input: examples
    output: probability of each unique class

    """
    if len(examples) == 0:
        return [0]
    class_data = examples[:,-1]
    class_data = list(class_data)
    probs = [class_data.count(num_classes[i]) for i in range(len(num_classes))]
    return np.asarray(probs) / len(class_data)


def information_gain(examples, attr_, threshold):
    """
    This fucntion will help us calcuate the information  gain if we select a certain attribute
    The calculations are based on Professor's notes.
    The use of  scipy to calculate entropy can be useful

    input: examples : the examples we have to match with out information gain
           attr_ : Out selected attribute for which we want to calculate entropy
           threshold : Our selected threshold

    output: gain - the total information gain.


    """
    left = examples[examples[:,attr_] < threshold]
    right = examples[examples[:,attr_] >= threshold]
    target_attr = list(examples[:,-1])

    dict_base = Counter(target_attr)
    target_attr = np.asarray(target_attr)
    target_attr = np.unique(target_attr)
    entropyBase = 0
    for i in range( len( target_attr ) ) :
        if dict_base[target_attr[i]] > 0:
            entropyBase = entropyBase - ( (dict_base[target_attr[i]]) / len( examples ) ) * math.log( ( dict_base[target_attr[i]] / len(examples) ), 2 )

    dict_left = Counter(left[:,-1])
    entropyLeft = 0
    target_left = np.asarray(left)
    target_left = np.unique(target_left)
    for i in range( len( target_left) ) :
        if dict_left[target_left[i]] > 0:
            entropyLeft = entropyLeft - ( (dict_left[target_left[i]]) / len( left) ) * math.log( ( (dict_left[target_left[i]]) / len( left ) ), 2 )

    dict_right = Counter(right[:,-1])
    target_right = np.asarray(right)
    target_right = np.unique(target_right)

    entropyRight = 0
    for i in range( len( target_right ) ) :
        if dict_right[target_right[i]] > 0:
            entropyRight = entropyRight - ( (dict_right[target_right[i]]) / len( right ) ) * math.log( (  (dict_right[target_right[i]]) / len( right ) ), 2 )

    gain = entropyBase - ( len( left ) / len( examples ) ) * entropyLeft - ( len( right ) / len( examples ) ) * entropyRight
    return gain


def choose_attr(examples, attributes):
    max_gain = best_attribute = best_threshold = -1
    if mode == 'optimized':
        for i in attributes:
            attribute_values = examples[:,i]
            L = min(attribute_values)
            M = max(attribute_values)
            for K in range(1,51):
                threshold = L + K*(M-L)/51
                gain = information_gain(examples, i, threshold)
                if gain > max_gain:
                    max_gain = gain
                    best_attribute = i
                    best_threshold = threshold
        return best_attribute, best_threshold, max_gain
    else:
        i =  random.randint(0,examples.shape[1]-2)
        attribute_values = examples[:,i]
        L = min(attribute_values)
        M = max(attribute_values)
        for K in range(1,51):
            threshold = L + K*(M-L)/51
            gain = information_gain(examples, i, threshold)
            if gain > max_gain:
                max_gain = gain
                best_attribute = i
                best_threshold = threshold
    return best_attribute, best_threshold, max_gain



def DTL(examples, attr, defaults):

    if examples.shape[0] < prun:
        full_tree = tree(-1,-1)
        full_tree.distribution = distribution(examples)
        return full_tree
    elif len(np.unique(examples[:,-1])) == 1:
        full_tree = tree(0,0)
        full_tree.distribution = [0 if i != examples[0,-1] else 1 for i in range(len(num_classes))]
        return full_tree
    else:
        best_attribute, best_threshold, max_gain = choose_attr(examples, attr)
        full_tree = tree(best_attribute, best_threshold)
        examples_left = examples[examples[:,best_attribute] < best_threshold]
        examples_right = examples[examples[:,best_attribute] >= best_threshold]
        #print(examples_left.shape)
        full_tree.gain = max_gain
        full_tree.left_child = DTL(examples_left, attr, distribution(examples))
        full_tree.right_child = DTL(examples_right, attr, distribution(examples))
    return full_tree




def DTL_TopLevel(examples):
    attr = [i for i in range(examples.shape[1]-1)]
    default  = distribution(examples)
    return DTL(examples, attr, default)


def get_class(test, trees, tree_id, node):
    #print(trees.left_child)
    print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n'%( tree_id, node+1, trees.best_attribute, trees.best_threshold, trees.gain));
    if trees.left_child is None and trees.right_child is None:
        dist = list(trees.distribution)
        predicted_classes = dist.index(max(dist)) + min(num_classes)
        return predicted_classes
    else:
        #print(trees.best_attribute)
        if test[trees.best_attribute] < trees.best_threshold or trees.right_child is None:
            return get_class(test, trees.left_child, tree_id, node+1)
        else:
            return get_class(test, trees.right_child, tree_id, node+1)


train_data = fetch_data(sys.argv[1])
num_classes = np.unique(train_data[:,-1])
test_data  = fetch_data(sys.argv[2])
mode = sys.argv[3]
prun = int(sys.argv[4])
forest = [DTL_TopLevel(train_data)]
#print(q)
if mode == 'forest3':
    for i in range(2):
        forest.append(DTL_TopLevel(train_data))
elif mode == 'forest15':
    for i in range(14):
        forest.append(DTL_TopLevel(train_data))
elif mode == 'optimized':
    pass
elif mode == 'randomized':
    pass
else:
    print("Please print out the correct option:")
    sys.exit(0)
final = []
for q in range(len(forest)):
    accuracy = []
    predicted_classes = [ get_class(i,forest[q],q+1,1) for i in test_data[:,:-1] ]
    for i in range(len(test_data)):
        if type(predicted_classes[i]) == np.float32 or type(predicted_classes[i]) == np.float64:
            if predicted_classes[i] == test_data[i,-1]:
                accuracy.append(1)
            else:
                accuracy.append(0)
        else:
            if test_data[i,-1] in predicted_classes[i]:
                accuracy.append(1/len(predicted_classes[i]))
            else:
                accuracy.append(0)
        print("ID=%5d"%(i+1)+ ", predicted=%3d"%(predicted_classes[i])+", true=%3d"%(test_data[i,-1])+", accuracy=%4.2f"%(accuracy[-1]))

    accuracy = np.asarray(accuracy)
    final.append(np.sum(accuracy) / len(accuracy) )
    print('classification accuracy=%6.4f'%( np.sum(accuracy) / len(accuracy) ))

print("Highest accuracy among trees is : %6.4f for tree %d "%(max(final), final.index(max(final))+1 ))
