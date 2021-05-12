import numpy as np
import math # just using the pow for calculating the norm

def KNN_test(X_train, Y_train, X_test, Y_test,K):
    NumberOfCorrect = 0
    for i, j in enumerate(X_test):
        radius = []
        for n, m in enumerate(X_train):
            radius.append(((math.pow((m[0] - j[0]), 2) + math.pow((m[1] - j[1]), 2)), Y_train[n][0]))
        List_sorted = sorted(radius, key=lambda member: member[0], reverse=False)[:K] # https://stackoverflow.com/questions/13669252/what-is-key-lambda/13669294
        predictedvalue = sum([member[1] for member in List_sorted])
        if (not predictedvalue):
            predictedvalue = List_sorted[0][1]
        elif (predictedvalue < 0):
            predictedvalue = -1
        elif (predictedvalue > 0):
            predictedvalue = 1
        if predictedvalue == Y_test[i][0]:
            NumberOfCorrect = NumberOfCorrect+1
    return NumberOfCorrect / len(X_test)

def choose_K(X_train,Y_train,X_val,Y_val):
    OldAccuracy = 0
    NewAccuracy = 0
    Predicted_K = 1
    K = len(X_train)
    for i in range(1, K):
        OldAccuracy = KNN_test(X_train,Y_train,X_val,Y_val, i)
        if(OldAccuracy>NewAccuracy):
            NewAccuracy = OldAccuracy
            Predicted_K = i
    return Predicted_K
