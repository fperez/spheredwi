import numpy as np
import regreg.api as rr
import pylab as pl
pl.ion()

"""
RegReg examples for minimizing the function

\|Y - X\beta\|_2^2 + \lambda_1 \|\beta\|_1 subject to \beta_i \geq 0.
"""



def example1(lambda1=10):

    #Example with X = np.identity(n)
    #Try varying lambda1 to see shrinkage

    n = 100

    Y = 10*np.random.standard_normal(n)

    loss = rr.l2normsq.shift(-Y,coef=1.)
    sparsity = rr.l1norm(n, lagrange = lambda1)
    nonnegative = rr.nonnegative(n)

    problem = rr.container(loss, sparsity, nonnegative)
    solver = rr.FISTA(problem)
    solver.fit(debug=True)

    solution = solver.composite.coefs

    pl.plot(Y, color='red', label='Y')
    pl.plot(solution, color='blue', label='beta')
    pl.legend()




def example2(lambda1=10):

    #Example with a non-identity X

    n = 100
    p = 1000

    X = np.random.standard_normal(n*p).reshape((n,p))
    Y = 10*np.random.standard_normal(n)

    loss = rr.l2normsq.affine(X,-Y,coef=1.)
    sparsity = rr.l1norm(p, lagrange = lambda1)
    nonnegative = rr.nonnegative(p)

    problem = rr.container(loss, sparsity, nonnegative)
    solver = rr.FISTA(problem)
    solver.fit(debug=True)

    solution = solver.composite.coefs



def example3(lambda1=10):

    #Example using a smooth approximation to the non-negativity constraint
    # On large problems this might be faster than using the actual constraint

    n = 100
    p = 1000


    X = np.random.standard_normal(n*p).reshape((n,p))
    Y = 10*np.random.standard_normal(n)

    loss = rr.l2normsq.affine(X,-Y,coef=1.)
    sparsity = rr.l1norm(p, lagrange = lambda1)
    nonnegative = rr.nonnegative(p)
    smooth_nonnegative = rr.smoothed_atom(nonnegative, epsilon = 1e-4)

    problem = rr.container(loss, sparsity, smooth_nonnegative)
    solver = rr.FISTA(problem)
    solver.fit(debug=True)

    solution1 = solver.composite.coefs



    loss = rr.l2normsq.affine(X,-Y,coef=1.)
    sparsity = rr.l1norm(p, lagrange = lambda1)
    nonnegative = rr.nonnegative(p)

    problem = rr.container(loss, sparsity, nonnegative)
    solver = rr.FISTA(problem)
    solver.fit(debug=True)

    solution2 = solver.composite.coefs


    pl.subplot(1,2,1)
    pl.hist(solution1, bins=40)

    pl.subplot(1,2,2)
    pl.scatter(solution2,solution1)
    pl.xlabel("Constraint")
    pl.ylabel("Smooth constraint")



def example4(lambda1=10):

    #Example with an initial value for backtracking

    # In the previous examples you'll see a lot of "Increasing inv_step" iterations - these are trying to find an approximate Lipschitz constant in a backtracking loop.
    # For your problem the Lipschitz constant is just the largest eigenvalue of X^TX, so you can precompute this with a few power iterations.

    n = 100
    p = 1000

    X = np.random.standard_normal(n*p).reshape((n,p))
    Y = 10*np.random.standard_normal(n)

    v = np.random.standard_normal(p)
    for i in range(10):
        v = np.dot(X.T, np.dot(X,v))
        norm = np.linalg.norm(v)
        v /= norm
    print "Approximate Lipschitz constant is", norm

    loss = rr.l2normsq.affine(X,-Y,coef=1.)
    sparsity = rr.l1norm(p, lagrange = lambda1)
    nonnegative = rr.nonnegative(p)

    problem = rr.container(loss, sparsity, nonnegative)
    solver = rr.FISTA(problem)

    #Give approximate Lipschitz constant to solver
    solver.fit(debug=True, start_inv_step=norm)

    solution = solver.composite.coefs
