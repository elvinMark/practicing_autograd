from autograd import grad
import autograd.numpy as np

# Defining some of the most common activation functions
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def tanh(x):
    return (1. - np.exp(-x)) / (1. + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def leaky_relu(x):
    return np.maximum(0.01*x , x)

# Defining a function to initialize our parameters
def init_params(size):
    N = np.prod(np.array(size))
    return 2. / N ** 0.5 *(np.random.random(size) - 0.5)

# Defining the forward operation for the neural network
"""
params will contain the parameters (weights and biases) of our simple multilayer perceptron.
params = [[w1,b1],[w2,b2],...]
"""
def neural_forward(params,x):
    o = x
    for w,b in params:
        o = sigmoid(np.dot(o,w) + b)
    return o

# Defining the loss function
def loss(params,x,y):
    o = neural_forward(params,x)
    return 0.5 * np.sum((o - y)**2)

# Calculating the gradient of the loss
dloss = grad(loss)

# Defining our input and output for the XOR problem
x = [[0.,0.],[1.,0.],[0.,1.],[1.,1.]]
y = [[1.,0.],[0.,1.],[0.,1.],[1.,0.]]

# Definig the parameters
w1 = init_params((2,3))
b1 = init_params((3,))
w2 = init_params((3,2))
b2 = init_params((2,))

params = [[w1,b1],[w2,b2]]

# Defining learning rate
lr = 0.7

# Training
for epoch in range(500):
    l = loss(params,x,y)
    dparams = dloss(params,x,y)
    print(dparams)safasdfasfdasfd
    
    print(f"epoch: {epoch}, loss: {l}")
    for (dw,db),(w,b) in zip(dparams,params):
        w -= lr * dw
        b -= lr * db

print(neural_forward(params,x))
