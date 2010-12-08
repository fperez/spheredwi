#Routines for solving penalized (either L1 or L2) minimization problems. Gamma is
#a tuning parameter. Need "from cvxmod import *"

def l1min(A,b,gamma):
  """Solve the l1 penalized minimization problem min||Ax-b||_2 + gamma||x||_1"""
  n = cols(A)
  x = optvar('x',n)
  p = problem(minimize(norm2(A*x - b) + gamma*norm1(x)))
  p.solve()
  return value(x)

def l2min(A,b,gamma):
  """Solve the l2 penalized minimization problem min||Ax-b||_2 + gamma||x||_2"""
  n = cols(A)
  x = optvar('x',n)
  p = problem(minimize(norm2(A*x - b) + gamma*norm2(x)))
  p.solve()
  return value(x)
