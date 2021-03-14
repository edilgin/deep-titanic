import numpy as np
import pandas as pd
#binary classification using a modular deep neural network


def loadData():
    # for ease of use we will read csv with pandas then use numpy to convert dataframes to np.arrays
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    trainData = pd.read_csv("titanic/train.csv")
    testData = pd.read_csv("titanic/test.csv")
    testData2 = pd.read_csv("titanic/gender_submission.csv")
    X_train = trainData[["Pclass","Sex","Age"]]
    Y_train = trainData[["Survived"]]
    X_test = testData[["Pclass", "Sex", "Age"]]
    Y_test = testData2[["Survived"]]
    #                 section of dropping values that are "NAN" from X and Y

    not_nans1 = X_train['Age'].notna()                 # mask we will apply to X_train and Y_train which is values that are not nan
    X_train = X_train[not_nans1]
    Y_train = Y_train[not_nans1]
    X_train.reset_index(drop=True, inplace=True)      # reseting indeces after dropping nan values
    Y_train.reset_index(drop=True, inplace=True)

    not_nans2 = X_test['Age'].notna()                 # mask we will apply to X_test and Y_test which is values that are not nan
    X_test = X_test[not_nans2]
    Y_test = Y_test[not_nans2]
    X_test.reset_index(drop=True, inplace=True)      # reseting indeces after dropping nan values
    Y_test.reset_index(drop=True, inplace=True)
    #                we will show gender with 1's and 0's because our model is only capable of making sense of numbers
    dict = {"female": 0, "male": 1}
    X_train = X_train.replace({"Sex":dict})
    X_test = X_test.replace({"Sex":dict})
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


def randomInit():
    W1 = np.random.randn(5,3) * 0.1
    W2 = np.random.randn(1,5) * 0.1
    b1 = np.zeros([1,5])
    b2 = np.zeros([1,1])
    parameters = {"W1": W1, "W2": W2,"b1": b1, "b2": b2}
    return parameters


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_prime(Z):
    f = 1 / (1 + np.exp(-Z))
    return f * (1-f)


def reLU(Z):
    return np.maximum(0,Z)


def reLU_prime(Z):
    return np.where(Z > 0, 1.0, 0.0)


def computeCost(A_last,Y):
    m = len(Y)
    cost = -(1/m) * (np.sum(np.log((A_last.T * Y)[A_last.T * Y != 0])) + np.sum(np.log(1 - (1 - Y) * A_last.T)))
    return cost


def forwardProp(X,parameters):
    W1, W2, b1, b2 = parameters["W1"], parameters["W2"], parameters["b1"], parameters["b2"]
    Z1 = np.dot(W1, X.T) + b1.T
    A1 = reLU(Z1)
    Z2 = W2.dot(A1) + b2.T
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "Z2": Z2, "A1": A1, "A2": A2}
    return A2, cache


def calcGrad(parameters,cache, X, Y):
    m = len(Y)
    W1, W2, b1, b2= parameters["W1"], parameters["W2"], parameters["b1"], parameters["b2"]
    A2, A1, Z2, Z1 = cache["A2"], cache["A1"], cache["Z2"], cache["Z1"]
    dZ2 = A2 - Y.T                      # dZ3 = (1,714)      A3 = (1,714)
    dW2 = (1/m) * dZ2.dot(A1.T)                      # dW3 = (1,5)        A2 = (5,714)
    db2 = (1/m) * np.array(([dZ2.sum(axis=1)]))      # db3 = (1,1)        A1 = (5,714)
    dZ1 = W2.T.dot(dZ2) * reLU_prime(Z1)   # dZ2 = (4,714)
    dW1 = (1/m) * dZ1.dot(X)
    db1 = (1/m) * np.array(([dZ1.sum(axis=1)]))
    grads = {"dW2": dW2, "db2": db2, "dW1": dW1, "db1": db1}
    return grads


def backwardProp(grads,parameters,learning_rate):
    dW2,db2,dW1,db1 = grads["dW2"],grads["db2"],grads["dW1"],grads["db1"]
    W1, W2,b1, b2 = parameters["W1"], parameters["W2"], parameters["b1"], parameters["b2"]
    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2
    parameters = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}
    return parameters


def predict(X_test,Y_test,parameters):
    prediction, cache = forwardProp(X_test,parameters)
    prediction = prediction.reshape(332,1)
    for i in range(len(prediction)):
        if prediction[i] >= 0.5:
            prediction[i] = 1
        else:
            prediction[i] = 0
    error = abs(prediction - Y_test)
    return sum(error)/len(error)


def neuralNetModelFirst(number_of_iterations=7000,learning_rate=0.04):
    X_train, Y_train, X_test, Y_test = loadData()
    parameters = randomInit()
    for i in range(0, number_of_iterations):
        prediction, cache = forwardProp(X_train, parameters)
        cost = computeCost(prediction, Y_train)
        grads = calcGrad(parameters, cache, X_train, Y_train)
        parameters = backwardProp(grads, parameters, learning_rate=learning_rate)
    results = predict(X_test, Y_test, parameters)
    return results


total_percantage = 0
for i in range(100):
    print(i)
    result = neuralNetModelFirst()
    total_percantage += result
print(total_percantage / 100)
