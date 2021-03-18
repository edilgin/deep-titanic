import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#binary classification using a modular deep neural network

def loadData():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    trainData = pd.read_csv("titanic/train.csv")
    testData = pd.read_csv("titanic/test.csv")
    testData2 = pd.read_csv("titanic/gender_submission.csv")
    X_train = trainData[["Pclass","Sex","Age","Fare"]]
    Y_train = trainData[["Survived"]]
    X_test = testData[["Pclass", "Sex", "Age","Fare"]]
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

    # lets normalize the age and fare values
    mean = np.mean(X_train["Age"])
    std_dev = np.std(X_train["Age"])
    X_train["Age"] = (X_train["Age"] - mean) / std_dev
    mean = np.mean(X_test["Age"])
    std_dev = np.std(X_test["Age"])
    X_test["Age"] = (X_test["Age"] - mean) / std_dev
    mean = np.mean(X_train["Fare"])
    std_dev = np.std(X_train["Fare"])
    X_train["Fare"] = (X_train["Fare"] - mean) / std_dev
    mean = np.mean(X_test["Fare"])
    std_dev = np.std(X_test["Fare"])
    X_test["Fare"] = (X_test["Fare"] - mean) / std_dev
    #                we will show gender with 1's and 0's because our model is only capable of making sense of numbers
    dict = {"female": 0, "male": 1}
    X_train = X_train.replace({"Sex":dict})
    X_test = X_test.replace({"Sex":dict})
    return np.array(X_train).T, np.array(Y_train).T, np.array(X_test).T, np.array(Y_test).T

def xaiverInit(layer_dims,activation_func):
    parameters = {}
    for i in range(1,len(layer_dims)):
        if activation_func[i-1] == "relu":
            activation = 2
        elif activation_func[i-1] == "sigmoid":
            activation = 1
        parameters["W" + str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1]) * np.sqrt(activation / layer_dims[i-1])
        parameters["b" + str(i)] = np.zeros([layer_dims[i],1])
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


def compute_cost(A_last,Y):
    m = len(Y.T)
    cross_ent_cost = -(1/m) * np.sum(Y * np.log(A_last) + (1-Y) * np.log(1 - A_last))
    return cross_ent_cost

def predict(X_test,Y_test,parameters,activation_funcs):
    cache = forward_prop(X_test,parameters,activation_funcs)
    prediction = cache["A" + str(len(cache)//2)]
    prediction.flatten()
    for i in range(len(prediction)):
        if prediction[0][i] >= 0.5:
            prediction[0][i] = 1
        else:
            prediction[0][i] = 0
    error = abs(prediction - Y_test)
    error = error[~np.isnan(error)]
    return np.sum(error) /len(error)

def forward_prop(X,parameters,activation_funcs):
    cache = {"A0": X}
    for i in range(1, (len(parameters)//2) + 1):
        cache["Z" + str(i)] = np.dot(parameters["W" + str(i)], cache["A" + str(i-1)]) + parameters["b" + str(i)]
        if activation_funcs[i-1] == "relu":
            cache["A" + str(i)] = reLU(cache["Z" + str(i)])
        elif activation_funcs[i-1] == "sigmoid":
            cache["A" + str(i)] = sigmoid(cache["Z" + str(i)])
    return cache

def calc_gradient(cache,parameters,activation_funcs,Y):
    m = len(Y)
    grads = {}
    index_of_last = len(parameters) // 2
    dA_last = -(Y/cache["A" + str(index_of_last)]) + ((1- Y)/(1-cache["A" + str(index_of_last)]))
    grads["dZ" + str(index_of_last)] = dA_last * sigmoid_prime(cache["Z" + str(index_of_last)])
    grads["dW" + str(index_of_last)] = (1/m) * np.dot(grads["dZ" + str(index_of_last)], cache["A" + str(index_of_last - 1)].T)
    grads["db" + str(index_of_last)] = (1/m) * np.sum(grads["dZ" + str(index_of_last)], axis=1, keepdims=True)
    for i in range((len(parameters)//2)-1, 0, -1):
            grads["dZ" + str(i)] = np.dot(parameters["W" + str(i+1)].T, grads["dZ" + str(i+1)]) * reLU_prime(cache["Z" + str(i)])
            grads["dW" + str(i)] = (1/m) * np.dot(grads["dZ" + str(i)], cache["A" + str(i-1)].T)
            grads["db" + str(i)] = (1/m) * np.sum(grads["dZ" + str(i)],axis=1,keepdims=True)
    return grads

def back_prop(grads,parameters,learning_rate):
    for i in range(1,(len(parameters) // 2)+1):
        parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate * grads["dW" + str(i)]
        parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate * grads["db" + str(i)]
    return parameters


def NeuralNet(layer_dims,activation_funcs,iterations,learning_rate):
    X_train, Y_train, X_test, Y_test = loadData()
    parameters = xaiverInit(layer_dims, activation_funcs)
    cost_list = []
    for i in range(1,iterations):
        cache = forward_prop(X_train, parameters,activation_funcs)
        cost = compute_cost(cache["A"+str(len(cache)//2)],Y_train)
        grads = calc_gradient(cache,parameters,activation_funcs,Y_train)
        parameters = back_prop(grads,parameters,learning_rate)
        cost_list.append(cost)
    hata = predict(X_test,Y_test,parameters,activation_funcs)
    print(hata)
    return cost_list
cost_list = NeuralNet([4,20,20,20, 1], ["relu", "relu","relu","sigmoid"],iterations=10000,learning_rate=0.00005)
X = []
for i in range(len(cost_list)):
    X.append(i)
plt.plot(X,cost_list)
plt.show()