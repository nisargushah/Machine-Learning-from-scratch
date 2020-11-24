import numpy as np
import pandas as pd
import sys
import random

class K_Means:
    def __init__(self, k=3, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            if sys.argv[2] == 'round_robin':
                self.centroids[i] = sum(data[0::i+1])/len(data[0::i+1])
                print("--------")
                print(self.centroids[i])
            elif sys.argv[2] == 'random':
                print("--------")
                self.centroids[i] = random.randint(int(np.amin(data)), int(np.amax(data)))
                print(self.centroids[i])
        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in df:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                #print(self.classifications)
                final = [1]*int(np.amax(data)+2)
                for x, y in self.classifications.items():
                    #final[y] = x
                    for value in y:
                        #print("Hey")
                        #print(value)
                        final[int(value[0])] = x +1
                    print(x, y)
                for val in data:
                    #print("Hello")
                    #print(val)
                    if(len(val)) == 2:
                        print('(%10.4f, %10.4f) --> cluster %d\n'% (val[0], val[1], final[int(val[0])]))
                    else:
                        print('%10.4f --> cluster %d\n'%( val, final[int(val)]))
                break


df = np.loadtxt('set1a.txt')
print(df[0,1])
#df.convert_objects(convert_numeric=True)


print(df)
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)
#y = [1,2,3,1,2,3,1,2]
clf = K_Means()
clf.fit(df)
