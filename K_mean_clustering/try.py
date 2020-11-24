import numpy as np
import sys
import random
from scipy.spatial import distance



def K_Means(data, k ,option):

    ##For this attmept we are assuming that its a round robin scene

    ##step 1 - read file

    k = int(k)
    ##Initliaze the centroidss
    oneC = data[:,0]
    twoC = data[:,1]
    #print(twoC)
    centroids = []
    for i in range(k):
        if option == 'round_robin':
            centroids.append( (sum(oneC[i::k])/len(oneC[i::k]),sum(twoC[i::k])/len(twoC[i::k])))

    #print(len(data), data[1])
    clusters = [1]*len(data)
    while True:
        cluster_copy = clusters.copy()
        for j in range(len(data)):
            #print(_data)
            dist = [distance.euclidean(i,data[j]) for i in centroids ]
            clusters[j] = dist.index(min(dist))+1

        for a in range(1,k+1):
            indices = [i for i, x in enumerate(clusters) if x == a]
            res_list = [data[i] for i in indices]
            #print(len(res_list))
            #sys.exit(0)
            oneC = [res_list[i][0] for i in range(len(res_list))]
            twoC = [res_list[i][1] for i in range(len(res_list))]
            len1 = len(oneC)
            len2 = len(twoC)
            if len1 == 0:
                len1 = 0.01
            if len2 == 0:
                len2 = 0.01


            centroids[a-1] = (sum(oneC)/len1, sum(twoC)/len2)

        if clusters == cluster_copy:
            break



    #print(centroids)
    for h in range(len(data)):
        print('(%10.4f, %10.4f) --> cluster %d'%( data[h][0], data[h][1], clusters[h]))






file = sys.argv[1]
data = np.loadtxt(file)
k = sys.argv[2]
option = sys.argv[3]

K_Means(data,k,option)
