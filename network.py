#http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np

# sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# input dataset
X = np.array([[4, 3, 6],
              [7, 11, 7],
              [3, 0, 81],
              [4, 21, 9]])

# output dataset
y = np.array([[4, 7, 3, 4]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(100)

# initialize weights randomly with mean 0
syn0 = 100 * np.random.random((3, 1)) - 1

for iter in xrange(10000):
    #forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    #how much did we miss?
    l1_error = y - l1

    #multiply how much we missed by the
    # #slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    #update weights
    syn0 += np.dot(l0.T, l1_delta)

print "Output After Training:"
print l1
print 'the end'
