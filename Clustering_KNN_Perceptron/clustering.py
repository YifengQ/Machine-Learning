import numpy as np
import math
import random

# aX = np.array([[0], [1], [2], [7], [8], [9], [12], [14], [15]])
# aK = 3
#
# bX = np.array([[1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2]])
# bK = 2

def Rand_Cluster_Centers(X,K):
    cluster_centers = []

    for j in range(K):
        cluster_center = []
        for i in range(X.shape[1]):
            cluster_dim = random.randint(min(X[:, i]), max(X[:, i]))
            cluster_center.append(cluster_dim)

        # Replay if duplicate cluster is found
        for cc in cluster_centers:
            if cc == cluster_center:
                return Rand_Cluster_Centers(X,K)
                break;

        cluster_centers.append(cluster_center)

    return cluster_centers

def Find_Distances(X, CC):
    all_distances = []

    for index, x in enumerate(X):

        dist_cluster_centers = []
        for index, cluster_center in enumerate(CC):
            add_dif_squared = 0

            for [ftr, dim] in zip(x, cluster_center):
                ftr_cnt_dif = ftr - dim
                add_dif_squared += ftr_cnt_dif ** 2

            dist_cluster_center = math.sqrt(add_dif_squared)
            dist_cluster_centers.append(dist_cluster_center)

        all_distances.append(dist_cluster_centers)

    return all_distances


def Build_Clusters(X, K, AD):
    clusters = []

    for i in range(K):
        cluster = []
        for [distances, x] in zip(AD, X):
            if distances.index(min(distances)) == i:
                cluster.append(x)

        clusters.append(cluster)
        #print('cluster: ', cluster)

    return clusters


def Calculate_Cluster_Centers(Clusters, Cluster_Centers, XShape1):
    new_cluster_centers = []

    for index, cluster in enumerate(Clusters):

        sum_dimensions = []
        avg_dimensions = []
        for i in range(XShape1):
            add_dim = 0
            for point in cluster:
                add_dim += point[i]

            sum_dimensions.append(add_dim)
            if len(cluster) == 0:
                avg_dimensions.append(Cluster_Centers[index][i])
            else:
                avg_dimensions.append(add_dim / len(cluster))

        new_cluster_centers.append(avg_dimensions)

    return new_cluster_centers


def K_Repeat(X, K, CC):
    all_distances = Find_Distances(X, CC)

    clusters = Build_Clusters(X, K, all_distances)

    xshape1 = X.shape[1]

    new_cluster_centers = Calculate_Cluster_Centers(clusters, CC, xshape1)
    #print('New_Cluster_Centers: ', new_cluster_centers)

    return new_cluster_centers


def K_Means(X,K):
    cluster_centers = Rand_Cluster_Centers(X, K)
    #cluster_centers = [[1], [8], [14]]
    #print('Original Cluster Centers: ', cluster_centers)

    new_cluster_centers = K_Repeat(X, K, cluster_centers)

    while cluster_centers != new_cluster_centers:
        cluster_centers = new_cluster_centers
        new_cluster_centers = K_Repeat(X, K, cluster_centers)

    return new_cluster_centers


def K_Means_better(X,K):
    resList = []
    iterations = 1000

    for i in range(iterations):
        fcc = K_Means(X, K)
        fcc.sort()
        resList.append(fcc)

    best_Count = 0
    best_Index = 0

    for i in range(iterations):
        count = resList.count(resList[i])
        if count > best_Count:
            best_Count = count
            best_Index = i

    return resList[best_Index]


# final_cluster_centers = K_Means(bX, bK)
# print('Final Cluster Centers: ', final_cluster_centers)
#
# print('Best Clusters Centers: ', K_Means_better(bX, bK))




