from cvxmod import *
import pylab
import routines.py

# Analyze a particular signal.
m = 50 # number of observations.
n = 200 # signal length.
s = 1.5e-2 # spike density.

# Generate a spike signal.
randseed(5)
x0 = zeros(n, 1)
for i in range(n):
    if rand() < s:
        x0[i] = 1

# Generate a random measurement matrix.
A = randn(m, n)

# Generate noisy measurements.
sigma = 0.1
v = sigma*randn(m, 1)
y = A*x0 + v

# Find signal reconstructions.
gamma = 0.01
x1 = l1min(A, y, gamma)
x2 = l2min(A, y, gamma)



