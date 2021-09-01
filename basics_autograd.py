import autograd.numpy as np
from autograd import grad

# Defining the function
def funf(x):
    return x - np.sin(x)

# Using grad to calculate the derivative of the function
dfunf = grad(funf)

# Using newton to find the solution of this non linear function

# Defining the initial value for the iterative method
x0 = 1. # It is important to make this number a float number and not an integer

# Iterate 10 times to find the aproximated solution using newton method
for i in range(10):
    x0 -= funf(x0)/dfunf(x0)
    print(f"x0 = {x0}, f(x0) = {funf(x0)}")


