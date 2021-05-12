# UNR CS Machine Learning
# Project 1 Decision Trees
# 9/26/19
# Students:
# Yifeng Qin
# Johan Yamssi

import numpy as np
import math

used = []
iteration = 0
previous_entropy = 0

#checks if there is division by zero, if there is return 0
def division_by_zero(x ,y):
    return x/y if y else 0

# gets the entropy of a feature / column
def getEntropyBase(Y):

    # declaration of counter variables
    num_of_true = 0
    num_of_false = 0
    num_of_rows = Y.shape[0]

    # for loop for the number of rows in X
    for i in range(num_of_rows):
        if Y[i] == 1:
            num_of_true += 1
        elif Y[i] == 0:
            num_of_false += 1

    # declare variables for the number of true or false over the total
    true_over_total = num_of_true / num_of_rows
    false_over_total = num_of_false / num_of_rows

    # checks so that we don't take the log of 0 which is infinity
    if true_over_total == 0 or false_over_total == 0:
        entropy = 0
    else:
        #equation for entropy
        entropy = -1 * (false_over_total * math.log2(false_over_total)) - (true_over_total * math.log2(true_over_total))

    return entropy


# returns the left and right values for a branch/feature
def branch_values(X, Y, feature):

    # size
    num_of_rows, num_of_cols = np.shape(X)

    # init a list to store the true and false lables corresponding to a feature
    true = []
    false = []

    # for loop for the number of rows in X
    for i in range(num_of_rows):

        if X[i][feature] == 1:
            true.append(Y[i])

        elif X[i][feature] == 0:
            false.append(Y[i])

    # gets the number of true and false values
    num_true_true = true.count(1)
    num_true_false = true.count(0)
    num_false_true = false.count(1)
    num_false_false = false.count(0)

    # total number of true and false
    total_true = len(true)
    total_false = len(false)

    # check for division by zero, then checks which proportion is bigger to set the right side by
    if division_by_zero(num_true_true,total_true) >= division_by_zero(num_true_false,total_true):
        value_true = 1
    else:
        value_true = 0

    # check for division by zero, then checks which proportion is bigger to set the left side by
    if division_by_zero(num_false_true,total_false) >= division_by_zero(num_false_false ,total_false):
        value_false = 1
    else:
        value_false = 0

    # want to get the entropy values for this feature
    entropy_false, entropy_true, test1, test2 = getEntropy(X, Y, feature)

    # checks which side has the highest entropy to split the tree
    if entropy_false <= entropy_true:
        split = 1
    else:
        split = 0

    return value_false, value_true, split

# gets the entropy of a feature
def getEntropy(X, Y, feature):

    # declaration of counter variables
    num_of_true_yes = 0
    num_of_true_no = 0
    num_of_false_yes = 0
    num_of_false_no = 0
    num_of_true = 0
    num_of_false = 0

    num_of_rows, num_of_cols = np.shape(X)
    # for loop for the number of rows in X
    for i in range(num_of_rows):

        if X[i][feature] == 1:

            num_of_true += 1 # counts total number of yes
            if Y[i] == 1:
                num_of_true_yes += 1
            else:
                num_of_true_no += 1

        elif X[i][feature] == 0:

            num_of_false += 1 # counts total number of no
            if Y[i] == 1:
                num_of_false_yes += 1
            else:
                num_of_false_no += 1

    # declare variables for the number of true or false over the total
    proportion_true_yes = division_by_zero(num_of_true_yes, num_of_true)
    proportion_true_no = division_by_zero(num_of_true_no, num_of_true)
    proportion_false_yes = division_by_zero(num_of_false_yes, num_of_false)
    proportion_false_no = division_by_zero(num_of_false_no,num_of_false)

    # the proportion of true or false values to the total number of samples
    true_over_total = num_of_true / num_of_rows
    false_over_total = num_of_false / num_of_rows

    # checks so we don't devide by zero
    if proportion_true_no == 0 or proportion_true_yes == 0:
        entropy_true = 0
    else:
        entropy_true = (-1 * (proportion_true_no * math.log2(proportion_true_no))) - (proportion_true_yes * math.log2(proportion_true_yes))

    # checks so we don't devide by zero
    if proportion_false_no == 0 or proportion_false_yes == 0:
        entropy_false = 0
    else:
        entropy_false = -1 * (proportion_false_no * math.log2(proportion_false_no)) - (proportion_false_yes * math.log2(proportion_false_yes))

    return entropy_false, entropy_true, false_over_total, true_over_total


# returns the information gain of a certain feature passed in
def getIG(X, Y, feature):

    global iteration
    global previous_entropy

    entropy_no, entropy_yes, proportion_root_no, proportion_root_yes = getEntropy(X, Y, feature)

    # the first iteration gets the entropy from the labels, then stores the entropy of decision we split on
    if iteration == 0:
        root_entropy = getEntropyBase(Y)
        previous_entropy = root_entropy
    else:
        root_entropy = previous_entropy

    # equation for information gain
    information_gain = root_entropy - ((proportion_root_no * entropy_no) + (proportion_root_yes * entropy_yes))

    return information_gain, entropy_no, entropy_yes, root_entropy


# returns the feature with the highest information gain
def bestIG(X, Y):

    global previous_entropy
    global iteration

    best = 0
    best_feature = 0
    best_entropy_no = 0
    best_entropy_yes = 0
    num_of_rows, num_of_cols = np.shape(X)

    # checks all the features
    for i in range(num_of_cols):
        #get the information gain from the feature passed in
        new, entropy_no, entropy_yes, root_e = getIG(X, Y, i)
        #checks if that information gain is greater then the last stored and that it hasen't been used already
        if new > best and i not in used:
            best_feature = i
            best = new
            best_entropy_no = entropy_no
            best_entropy_yes = entropy_yes

    # appeds to used to store all the used features
    used.append(best_feature)
    # gets the labels for the left and right child, split is which way the tree goes
    best_false, best_true, best_split = branch_values(X, Y, best_feature)

    # checks which side the tree will add the next branch to store the entropy of the previous decision
    if best_split > 0:
        previous_entropy = best_entropy_yes

    else:
        previous_entropy = best_entropy_no

    # height of the tree
    iteration += 1

    return best_feature, best_false, best_true, best_split, best


# returns how many the number of true of false values in a feature still
def in_feature(X, feature, expression):

    count = 0
    for k in range(X.shape[0]):
        if X[k][feature] != expression:
            count += 1

    return count

# returns the feature and label arrays without the unneeded decisions
def find_rows(X, Y, feature, expression):

    # deleltes the side of the tree that already has a decision
    while in_feature(X, feature, expression):

        for i in range(X.shape[0]):
            if X[i][feature] != expression:
                X = np.delete(X, i, axis = 0)
                Y = np.delete(Y, i, axis = 0)
                break

    return X, Y

# returns a list of lists of decision tree
def DT_train_binary(X,Y,max_depth):

    global iteration
    num_of_rows, num_of_cols = np.shape(X)
    IG = 1
    # runs for the number max depth
    if max_depth == -1:
        depth = num_of_cols

    else:
        depth = max_depth

    tree = []
    for i in range(depth):
        feature, value_false, value_true, split, IG = bestIG(X, Y)

        X, Y = find_rows(X, Y, feature, split)

        # if end of tree, the split value will be set to -1 for later processing
        if i == depth - 1:
            split = -1
        # if the IG is zero then there is nothing more to gain
        if IG == 0:
            split = -1
            tree.append([feature, value_false, value_true, split])
            iteration = 0
            used.clear()
            return tree
        else:
            tree.append([feature, value_false, value_true, split])

    iteration = 0
    used.clear()
    return tree


# returns the answer from the build decsion tree based on a test sample
def DT_Traverse(X_train, X_test):

    sizeDT = len(X_test)
    idx = 0

    #runs for the number of features
    while idx < sizeDT:
        # this gets the feature needed to be tested
        tree_feature = X_train[idx][0]

        # if the feature that is tested equals 0
        if X_test[tree_feature] == 0:
            # if it is on the opposite side it branches, that means it the end and returns the branch value, -1 means end of tree
            if X_train[idx][3] == 1 or X_train[idx][3] == -1:
                return X_train[idx][1]
            #if it is on the branch side, it will go to the next feature to be tested
            else:
                idx += 1
        else:
            if X_train[idx][3] == 0 or X_train[idx][3] == -1:
                return X_train[idx][2]
            else:
                idx += 1

def DT_test_binary(X,Y,DT):

    num_samples, test_cols = np.shape(X)
    correct = 0

    # checks if the label is
    for i in range(num_samples):
        ans = DT_Traverse(DT, X[i])
        if ans == Y[i]:
            correct += 1

    # gets the percentage
    accuracy = (correct / num_samples) * 100

    return accuracy


def DT_train_binary_best(X_train, Y_train, X_val, Y_val):

    num_samples, features = np.shape(X_train)
    accuracy = 0
    best_tree = []
    for i in range(features):

        # gets the tree with height based on the index
        new_tree = DT_train_binary(X_train, Y_train, i + 1)
        # gets the accuracy of the tree just created
        new_accuracy = DT_test_binary(X_val, Y_val, new_tree)

        if accuracy < new_accuracy:
            accuracy = new_accuracy
            best_tree = new_tree

    return best_tree

def DT_make_prediction(x,DT):
    return DT_Traverse(DT,x)

#############################################
#Split between real and binary functions####
############################################

# returns the average number of the features
def average_of_features(X, feature):

    # size
    num_of_rows, num_of_cols = np.shape(X)

    total = 0

    # for loop for the number of rows in X
    for i in range(num_of_rows):
        total += X[i][feature]

    avg = total / num_of_rows

    return avg


def branch_values_real(X, Y, feature):

    num_of_rows, num_of_cols = np.shape(X)

    avg = average_of_features(X, feature)

    under_label = []
    over_label = []

    for i in range(num_of_rows):
        if X[i][feature] >= avg:
            over_label.append(Y[i])
        else:
            under_label.append(Y[i])


    # gets the number of true and false values
    num_true_true = over_label.count(1)
    num_true_false = over_label.count(0)
    num_false_true = under_label.count(1)
    num_false_false = under_label.count(0)

    # total number of true and false
    total_true = len(over_label)
    total_false = len(under_label)

    # check for division by zero, then checks which proportion is bigger to set the right side by
    if division_by_zero(num_true_true, total_true) >= division_by_zero(num_true_false, total_true):
        value_true = 1
    else:
        value_true = 0

    # check for division by zero, then checks which proportion is bigger to set the left side by
    if division_by_zero(num_false_true, total_false) >= division_by_zero(num_false_false, total_false):
        value_false = 1
    else:
        value_false = 0

    # want to get the entropy values for this feature
    entropy_false, entropy_true, test1, test2 = getEntropy_real(X, Y, feature)

    # checks which side has the highest entropy to split the tree
    if entropy_false <= entropy_true:
        split = 1
    else:
        split = 0

    return value_false, value_true, split, avg

# gets the entropy of a feature
def getEntropy_real(X, Y, feature):

    # declaration of counter variables
    num_of_true_yes = 0
    num_of_true_no = 0
    num_of_false_yes = 0
    num_of_false_no = 0
    num_of_true = 0
    num_of_false = 0

    num_of_rows, num_of_cols = np.shape(X)

    avg = average_of_features(X, feature)

    # for loop for the number of rows in X
    for i in range(num_of_rows):

        if X[i][feature] >= avg:

            num_of_true += 1 # counts total number of yes
            if Y[i] == 1:
                num_of_true_yes += 1
            else:
                num_of_true_no += 1

        else:

            num_of_false += 1 # counts total number of no
            if Y[i] == 1:
                num_of_false_yes += 1
            else:
                num_of_false_no += 1

    # declare variables for the number of true or false over the total
    proportion_true_yes = division_by_zero(num_of_true_yes, num_of_true)
    proportion_true_no = division_by_zero(num_of_true_no, num_of_true)
    proportion_false_yes = division_by_zero(num_of_false_yes, num_of_false)
    proportion_false_no = division_by_zero(num_of_false_no,num_of_false)

    # the proportion of true or false values to the total number of samples
    true_over_total = num_of_true / num_of_rows
    false_over_total = num_of_false / num_of_rows

    # checks so we don't divide by zero
    if proportion_true_no == 0 or proportion_true_yes == 0:
        entropy_true = 0
    else:
        entropy_true = (-1 * (proportion_true_no * math.log2(proportion_true_no))) - (proportion_true_yes * math.log2(proportion_true_yes))

    # checks so we don't divide by zero
    if proportion_false_no == 0 or proportion_false_yes == 0:
        entropy_false = 0
    else:
        entropy_false = -1 * (proportion_false_no * math.log2(proportion_false_no)) - (proportion_false_yes * math.log2(proportion_false_yes))

    return entropy_false, entropy_true, false_over_total, true_over_total


# returns the information gain of a certain feature passed in
def getIG_real(X, Y, feature):

    global iteration
    global previous_entropy

    entropy_no, entropy_yes, proportion_root_no, proportion_root_yes = getEntropy_real(X, Y, feature)

    # the first iteration gets the entropy from the labels, then stores the entropy of decision we split on
    if iteration == 0:
        root_entropy = getEntropyBase(Y)
        previous_entropy = root_entropy
    else:
        root_entropy = previous_entropy

    # equation for information gain
    information_gain = root_entropy - ((proportion_root_no * entropy_no) + (proportion_root_yes * entropy_yes))

    #print(information_gain)

    return information_gain, entropy_no, entropy_yes, root_entropy


# returns the feature with the highest information gain
def bestIG_real(X, Y):

    global previous_entropy
    global iteration

    best = 0
    best_feature = 0
    best_entropy_no = 0
    best_entropy_yes = 0
    num_of_rows, num_of_cols = np.shape(X)

    # checks all the features
    for i in range(num_of_cols):
        #get the information gain from the feature passed in
        new, entropy_no, entropy_yes, root_e = getIG_real(X, Y, i)
        #checks if that information gain is greater then the last stored and that it hasen't been used already
        if new > best and i not in used:
            best_feature = i
            best = new
            best_entropy_no = entropy_no
            best_entropy_yes = entropy_yes

    # appends to used to store all the used features
    used.append(best_feature)
    # gets the labels for the left and right child, split is which way the tree goes
    best_false, best_true, best_split, avg = branch_values_real(X, Y, best_feature)

    # checks which side the tree will add the next branch to store the entropy of the previous decision
    if best_split > 0:
        previous_entropy = best_entropy_yes

    else:
        previous_entropy = best_entropy_no

    # height of the tree
    iteration += 1

    return best_feature, best_false, best_true, best_split, best, avg

# returns how many the number of true of false values in a feature still
def in_feature_real(X, feature, expression, avg):

    count = 0
    for k in range(X.shape[0]):
        if expression == 0 and X[k][feature] > avg:
            count += 1
        elif expression == 1 and X[k][feature] <= avg:
            count += 1

    return count

# returns the feature and label arrays without the unneeded decisions
def find_rows_real(X, Y, feature, expression, avg):

    # deletes the side of the tree that already has a decision
    while in_feature_real(X, feature, expression, avg):

        for i in range(X.shape[0]):

            if expression == 0 and X[i][feature] > avg:
                X = np.delete(X, i, axis=0)
                Y = np.delete(Y, i, axis=0)
                break
            elif expression == 1 and X[i][feature] <= avg:
                X = np.delete(X, i, axis = 0)
                Y = np.delete(Y, i, axis = 0)
                break

    return X, Y

# returns a list of lists of decision tree
def DT_train_real(X,Y,max_depth):

    global iteration
    num_of_rows, num_of_cols = np.shape(X)

    # runs for the number max depth
    if max_depth == -1:
        depth = num_of_cols

    else:
        depth = max_depth

    tree = []
    for i in range(depth):
        feature, value_false, value_true, split, IG, avg = bestIG_real(X, Y)

        X, Y = find_rows_real(X, Y, feature, split, avg)

        # if end of tree, the split value will be set to -1 for later processing
        if i == depth - 1:
            split = -1
        # if the IG is zero then there is nothing more to gain
        if IG == 0:
            split = -1
            tree.append([feature, value_false, value_true, split, avg])
            iteration = 0
            used.clear()
            return tree
        else:
            tree.append([feature, value_false, value_true, split, avg])

    iteration = 0
    used.clear()
    return tree

# returns the answer from the build decsion tree based on a test sample
def DT_Traverse_real(X_train, X_test):

    sizeDT = len(X_train)
    idx = 0

    #runs for the number of features
    while idx < sizeDT:
        # this gets the feature needed to be tested
        tree_feature = X_train[idx][0]

        # if the feature that is tested equals 0
        if X_test[tree_feature] <= X_train[idx][4]:
            # if it is on the opposite side it branches, that means it the end and returns the branch value, -1 means end of tree
            if X_train[idx][3] == 1 or X_train[idx][3] == -1:
                return X_train[idx][1]
            #if it is on the branch side, it will go to the next feature to be tested
            else:
                idx += 1
        else:
            if X_train[idx][3] == 0 or X_train[idx][3] == -1:
                return X_train[idx][2]
            else:
                idx += 1

def DT_test_real(X,Y,DT):

    num_samples, test_cols = np.shape(X)
    correct = 0

    # checks if the label is
    for i in range(num_samples):
        ans = DT_Traverse_real(DT, X[i])
        if ans == Y[i]:
            correct += 1

    # gets the percentage
    accuracy = (correct / num_samples) * 100

    return accuracy


def DT_train_real_best(X_train,Y_train,X_val,Y_val):

    num_samples, features = np.shape(X_train)
    accuracy = 0
    best_tree = []
    for i in range(features):

        # gets the tree with height based on the index
        new_tree = DT_train_binary(X_train, Y_train, i + 1)
        # gets the accuracy of the tree just created
        new_accuracy = DT_test_binary(X_val, Y_val, new_tree)

        if accuracy < new_accuracy:
            accuracy = new_accuracy
            best_tree = new_tree

    return best_tree
