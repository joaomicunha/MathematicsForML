

# Lagrange Optimizer for function f(x) constrained on g(x): 

# Import libraries
import numpy as np
from scipy import optimize

def f (x, y) :
    return -np.exp(x + y**2 - x*y)

def g (x, y) :
    return np.cosh(y) + x - 2

def dfdx (x, y) :
    return -(1+y)*np.exp(x-y**2+x*y)

def dfdy (x, y) :
    return -(-2*y + x)*np.exp(x-y**2+x*y)

def dgdx (x, y) :
    return 1

def dgdy (x, y) :
    return np.sinh(y)


def DL (xyλ) :
    [x, y, λ] = xyλ
    return np.array([
            dfdx(x, y) - λ * dgdx(x, y),
            dfdy(x, y) - λ * dgdy(x, y),
            - g(x, y)
        ])

# Run the optimizer from scipy:
(x0, y0, λ0) = (1,0, 0)
x, y, λ = optimize.root(DL, [x0, y0, λ0]).x

print("x = %g" % x)
print("y = %g" % y)
print("λ = %g" % λ)
print("f(x, y) = %g" % f(x, y))


