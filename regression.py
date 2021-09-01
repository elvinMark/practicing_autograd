from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Defining the model function
def model_fun(params,x):
    m,b = params
    return m*x + b

# Defining the loss function for the regression
def loss(params,x,y):
    o = model_fun(params,x)
    return np.sum((o - y)**2) / len(y)

# Gradient of the loss function
dloss = grad(loss)

# Generating the data
x = np.linspace(0,3,100)
y = 12*x + 8 + 7*(np.random.random(100) - 0.5)

# Initialize the parameters
params = np.array([2.,2.])

for i in range(50):
    l = loss(params,x,y)
    dparams = dloss(params,x,y)
    params -= 0.01*dparams
    plt.clf()
    plt.grid()
    plt.title("Linear Regression")
    plt.text(0,40,"m=%.3f, b=%.3f"%(params[0], params[1]))
    plt.text(0,35,"loss=%.3f"%l)
    plt.scatter(x,y,color='r')
    plt.plot(x,model_fun(params,x))
    plt.pause(0.05)

plt.show()
