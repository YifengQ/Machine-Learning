from nearest_neighbors import *
from perceptron import *
from clustering import *

# Data for KNN

XT = np.array([[1, 5], [2, 6], [2, 7], [3, 7], [3, 8], [4, 8], [5, 1], [5, 9], [6, 2], [7, 2], [7, 3], [8, 3], [8, 4], [9, 5]])
YT = np.array([[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]])

X = np.array ([[1, 1], [2, 1], [0, 10], [10, 10], [5, 5], [3, 10], [9, 4], [6, 2], [2, 2], [8, 7]])
Y = np.array( [[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])

# Data for KMeans
KMeansX = np.array([[1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2]])
# KMeansX = np.array([[0], [1], [2], [7], [8], [9], [12], [14], [15]])

# Data for Perceptron
Perceptron_X1 = np.array([[0, 1], [1, 0], [5, 4], [1, 1], [3, 3], [2, 4], [1, 6]])
Perceptron_Y1 = np.array([[1], [1], [0], [1], [0], [0], [0]])
Perceptron_X2 = np.array([[-2, 1], [1, 1], [1.5, -0.5], [-2, -1], [-1, -1.5], [2, -2]])
Perceptron_Y2 = np.array([[1], [1], [1], [-1], [-1], [-1]])


def main():
    print("-------------------- Part 1 ----------------------------\n")
    print("KNN_Test:", KNN_test(XT,YT,X, Y, 1))
    print("Chosen_K:", choose_K(XT,YT,X,Y))
    print()

    print("-------------------- Part 2 ----------------------------\n")
    print("KMeans:", K_Means(KMeansX, 2))
    print("KMeansBetter:", K_Means_better(KMeansX, 2))
    print()

    print("-------------------- Part 3 ----------------------------\n")
    print("Perceptron_Train:",perceptron_train(Perceptron_X2 , Perceptron_Y2))  # for test save these into w
    W = perceptron_train(Perceptron_X2 , Perceptron_Y2)
    print("Perceptron_Test:",perceptron_test(Perceptron_X2, Perceptron_Y2, W[0], W[1]))
    print()


if __name__ == "__main__":
    main()