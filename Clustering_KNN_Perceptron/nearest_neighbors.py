import numpy as np
import math

# test_X = np.array([[1, 1], [2, 1], [0, 10], [10, 10], [5, 5], [3, 10], [9, 4], [6, 2], [2, 2], [8, 7]])
# test_Y = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])
#
# train_X = np.array([[1, 5], [2, 6], [2, 7], [3, 7], [3, 8], [4, 8], [5, 1], [5, 9], [6, 2], [7, 2], [7, 3], [8, 3],
#                     [8, 4], [9, 5]])
# train_Y = np.array([[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]])
#
# test3_X = np.array([[1, 1,1], [2, 1,1], [0, 10,1], [10, 10,1], [5, 5,1], [3, 10,1], [9, 4,1], [6, 2,1], [2, 2,1], [8, 7,1]])
# train3_X = np.array([[1, 5,1], [2, 6,1], [2, 7,1], [3, 7,1], [3, 8,1], [4, 8,1], [5, 1,1], [5, 9,1], [6, 2,1], [7, 2,1], [7, 3,1], [8, 3,1],
#                     [8, 4,1], [9, 5,1]])

# k_Val = 5
#
# one_X = np.array([[2, 8]])
# one_Y = np.array([[1]])


def Find_Distances(X, point):
    all_distances = []
    point = np.array([point])

    for x in X:

        dist_points = []
        for p in point:
            add_dif_squared = 0

            for [ftr, dim] in zip(x, p):
                ftr_cnt_dif = ftr - dim
                add_dif_squared += ftr_cnt_dif ** 2

            dist_point = math.sqrt(add_dif_squared)
            dist_points.append(dist_point)

        all_distances.append(dist_points)

    return all_distances


def KNN_test(X_train, Y_train, X_test, Y_test, K):

    count_matches = 0
    for test, label in zip(X_test, Y_test):
        dist = Find_Distances(X_train, test)
        paired = np.array(list(zip(dist, Y_train)))
        paired_sorted = sorted(paired, key=lambda x: x[0])
        add_y = 0
        for i in range(K):
            add_y += paired_sorted[i][1]

        if add_y > 0:
            if label > 0:
                count_matches += 1

        else:
            if not label > 0:
                count_matches += 1

    accuracy = count_matches/len(Y_test)

    return accuracy


def choose_K(X_train, Y_train, X_val, Y_val):
    stop = len(Y_train)
    best_accuracy = 0
    best_k = 0
    for k in range(1, stop, 2):
        accuracy = KNN_test(X_train, Y_train, X_val, Y_val, k)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    return best_k



# accuracy = KNN_test(train_X, train_Y, test_X, test_Y, k_Val)
#
# print('accuracy: ', accuracy)
#
# bk = choose_K(train_X, train_Y, test_X, test_Y)
#
# print('best k: ', bk)
