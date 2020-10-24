import os
import numpy as np
import collections
import math
from scipy.stats import entropy


def fetch_data(filename):
    """
    In this function we will be taking in the filename(with relative location) and then
    sperate them to data points or attributes and class labels

    input: filename
    output : X - attributes (with data points)
             y - class labels
    """
    data = np.loadtxt(filename, dtype=np.float32)
    y    = data[:,-1]
    X    = data[:,:-1]

    return X, y


def distribution(class_labels):
    """
    This function will be used to find probability of classes in the examples (train or test)

    inputs: class_labels

    outputs: class_prob (array with class probability)

    """
    freq = collections.Counter(class_labels)
    class_prob = [value/len(class_labels) for (key,value) in freq.items()]

    return np.array(class_prob)



def information_gain(examples,classes, attr, threshold):
    pass
