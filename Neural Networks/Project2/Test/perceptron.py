import numpy as np

def perceptron_train(X,Y):
    Done = False # converged or not 
    epoch = 0 #keep track of epoch in case the data is not linearly separable
    maxEpoch = 199
    activation = 0 # initial activation
    W_B_List = [[],[]] # 0 is weight, 1 is bias 
    W_B_List[0] =[0]*len(X[0]) #initialize to 0
    W_B_List[1] = [0]
    while ( not Done or epoch < maxEpoch): # change the number of epochs to ur desire 
        epoch+=1
        for i in range (len(X)):
            activation = 0
            for m in range(len(X[i])):
                activation += X[i][m] * W_B_List[0][m]
            activation+= W_B_List[1][0]
            if  (( activation * Y[i][0] ) <= 0):
                for j in range(len(X[0])):
                    W_B_List[0][j] = W_B_List[0][j] + Y[i][0] * X[i][j]
                W_B_List[1][0] = W_B_List[1][0] + Y[i][0]
            else:
                Done = True # if it is not going to update anymore change the value and get the hell out
    return W_B_List

def perceptron_test(X_test, Y_test, w, b):
    NumberOfCorrect = 0
    activation = 0
    for i, j in enumerate(X_test):
        activation = np.dot(j,w) + b # could've done the above using the built in function, found out late!
        if activation * Y_test[i] > 0:
            NumberOfCorrect+=1
    return NumberOfCorrect/len(X_test)
