import numpy as np
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
"""
The algorithms in the book were straight forward but when implemented they seemed really ineffieient 
so I used this article http://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf to implement 
some of the functions using their approach
"""
"""
calculate a, h, z and return only z since we need z to calculate y_hat
"""
def calculate_helper_variables(model,X, h_flag = False):
    a = np.dot(X, model["W1"]) + model["b1"]
    h = np.tanh(a)
    z = np.dot(h, model["W2"]) + model["b2"]
    if h_flag:
        return z,h
    return z
def calculate_loss(model, X, y):
    L = 0
    radius = y.shape[0]
    z = calculate_helper_variables(model,X)
    y_hat = np.exp(z) / (np.sum(np.exp(z), axis=1)).reshape(-1, 1)
    for i in range(radius):
        L = L + np.log(y_hat[i][y[i]])
    return ((-1/radius) * (L))
def predict(model, x):
    Prediction = []
    z = calculate_helper_variables(model,x)
    y_hat = np.exp(z - np.max(z)) / (np.exp(z - np.max(z))).sum()
    for i in y_hat:
        Prediction.append(np.argmax(i))
    return np.array(Prediction)
"""
One hot encoding: We are using this process so we can convert our categorical variables into a form that could be provided to the ML algorithms 
so we can do a better job in prediction and make the algorithm possibly faster (in our case a lot) since we do not have to do that at each epoch and iteration
of our build model. 
"""
# The below algorithm was actually taken from 
# https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
"""
You can think of this as having 
[Coffee, Tea, Tea, Coffee]
if we do integer encoding we can represent it as 
[1, 0, 0, 1]
now with one hot encoding we get 
[ [0,1],
  [1,0],
  [1,0],
  [0,1] ]
The basic idea is that  we need a mechanism to somehow revert the output to the same format 
as that of the input data. This is why we are using One-Hot Encoding
"""
def _one_hot_values(labels_data):
    encoded = [0] * len(labels_data)
    for j, i in enumerate(labels_data):
        max_value = [0] * (np.max(labels_data) + 1)
        max_value[i] = 1
        encoded[j] = max_value
    return np.array(encoded)
def randomInitializer(y, nn_hdim):
    W1 = np.random.rand(2, nn_hdim)
    b1 = np.random.rand(1,nn_hdim)
    W2 = np.random.rand(nn_hdim, 2)
    b2 = np.random.rand(1,2)
    return W1,b1,W2,b2
def updateModel(model, dLdb1, dLdb2, dLdW1, dLdW2, eta=0.01):
    model["b1"] = model["b1"] - eta * (dLdb1)
    model["b2"] = model["b2"] - eta * (dLdb2)
    model["W1"] = model["W1"] - eta * (dLdW1)
    model["W2"] = model["W2"] - eta * (dLdW2)
def CalculateGradient(model , y_hat, ylabel, h ):
    dL_dyhat =  y_hat - ylabel
    dLda = (1 - (pow(h,2))) * (np.dot(dL_dyhat,model["W2"].transpose()))
    dLdb1 = np.sum(dLda)
    dLdW1 = np.dot(X.transpose(),dLda)
    dLdb2 = np.sum(dL_dyhat)
    dLdW2 = np.dot(h.transpose(),dL_dyhat)
    return dLdb1,dLdW1,dLdb2,dLdW2
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    W1,b1,W2,b2 = randomInitializer(y, nn_hdim)
    model = {"W1" : W1, "W2": W2, "b1":b1, "b2":b2}
    ylabel = _one_hot_values(y)
    if (print_loss):
        print("New HiddenLayerSize")
    for index in range(1,num_passes):
        z,h = calculate_helper_variables(model,X,True)
        y_hat = np.exp(z) / (np.sum(np.exp(z), axis=1, keepdims=True))
        dLdb1,dLdW1,dLdb2,dLdW2 = CalculateGradient(model , y_hat, ylabel, h)
        updateModel(model, dLdb1, dLdb2, dLdW1, dLdW2)
        if (print_loss) and  not (index % 1000):
            print("iteration:", index, "Loss:", calculate_loss(model, X, y))
    return model
"""
The below is just what was given to us in the project description
"""
"""
def plot_decision_boundary(pred_func, X, y):
    # set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
# Generate a dataset and plot it 
np.random.seed(0)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
# Generate ouputs with the following code 
plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('HiddenLayerSize%d' % nn_hdim)
    model = build_model(X, y, nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.show()
"""


