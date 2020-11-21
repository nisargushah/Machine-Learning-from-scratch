import numpy as np
import sys
import warnings
import random
from statistics import mean

class K_mean:

    def __init__(self, data, k, initialization):
        self.data = data
        self.K = k
        self.initialization = initialization



    def run(self, data):

        if self.K > len(data):
            warnings.warn("K cannot be greater than equal to the total length of the data, please select a different value for k")
            sys.exit(0)

        ##Step 1 assigning random initialization.
        self.centroid=[]
        for j in range(self.K):
            self.centroid.append(random.randint(min(data),max(data)))

        ##Step2 assigning the cluster number to all data points:
        if initialization == "random":
            self.cluster = []
            for j in range(len(data)):
                self.cluster.append(random.randint(1,self.K))
        elif initialization =="round_robin":
            self.cluster = []
            j=0
            count= 1
            while j < len(data):
                if count > self.K:
                    count = 1
                self.cluster.append(count)
                count+=1
                j+=1

        else:
            warnings.warn("Please enter a valid value for initialization (random or round_robin) ")
            sys.exit(0)

        ##Step 3, find distance of each element from the cluster
        print(self.cluster)
        print(self.centroid)

if __name__ == "__main__":
    #s<data_file>, <K>, <initialization>
    data = np.loadtxt(sys.argv[1])
    #print(len(data))
    k = sys.argv[2]
    initialization = sys.argv[3]

    clf = K_mean(data, int(k) , initialization)
    clf.run(data)
