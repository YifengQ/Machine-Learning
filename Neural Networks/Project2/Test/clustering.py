import numpy as np
import math
import random

# X = np.array([[1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2]])
# X = np.array([[0], [1], [2], [7], [8], [9], [12], [14], [15]])

def K_Means(X, K):

    # Randomly initialize center
    randCenterIndexes = random.sample(range(len(X)), K)
    randCenters = []
    for k in randCenterIndexes:
        randCenters.append(X[k])

    output = FindClusters(X, randCenters)

    return output

def FindClusters(X, centers):
    oldCenters = []
    error = 100


    while error != 0:
        clusters = [[] for i in range(len(centers))]
        for i in range(len(X)):
            temp = 5444654
            for centerIndex in range(len(centers)):
                dist = calcDistance(X[i], centers[centerIndex])

                if (dist <= temp):
                    temp = dist
                    value = X[i]
                    found = centerIndex

            # Store value in corresponding cluster (i.e found)
            clusters[found].append(value)

        oldCenters = centers.copy()

        # Jank but to dynamically make the dimensions, e.g: x -- x,y -- x,y,z etc.
        # And to store the new centers again
        dimensions = [[] for i in range(clusters[0][0].shape[0])]
        centers = [[] for i in range(len(centers))]

        # Iterate over the cluster and store the values in their respective array
        # depending on the dimensions noted above
        for i in range(len(centers)):
            for j in clusters[i]:
                for val in range(len(j)):
                    dimensions[val].append(j[val])

            # Iterate over the dimension list and calculate the mean for that cluster
            # To get the new cluster center
            for numDim in range(len(dimensions)):
                value = np.mean(dimensions[numDim])
                centers[i].append(value)

        try:
            if (np.allclose([centers], [oldCenters])):
                # print("Clusters:", clusters)
                # print("Centers:", centers)
                error = 0
        except:
            pass

        dimensions.clear()

    # print(clusters)
    return centers


def calcDistance(point, centerPoint):
    distance = 0
    for i in range(point.shape[0]):
        value = (centerPoint[i] - point[i])**2
        distance += value

    distance = math.sqrt(distance)

    return distance


def K_Means_better(X, K):
    run = 1000
    bestCenterList = []
    occurences = []

    j = 0
    for i in range(run):
        try:
            result = K_Means(X, K)
            bestCenterList.append(result)
            # print(result)
            j += 1
        except:
            pass

    for j in range(len(bestCenterList)):
        count = bestCenterList.count(bestCenterList[j])
        occurences.append(count)

    maxNum = 0

    for k in occurences:
        if (k > maxNum):
            maxNum = k

    # print(j)
    # print(maxNum)
    # print(bestCenterList[maxNum])

    return bestCenterList[maxNum]


# print(K_Means(X, 3))
# print(K_Means_better(X, 3))