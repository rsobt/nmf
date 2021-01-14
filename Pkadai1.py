from numpy import *
from nmf import *
A=matrix([[1,1,2,1,3],[2,3,3,4,4],[1,1,2,1,3],[1,2,1,3,1]])
U,V=factorize(A,dim=2,iteration_num=100,seed=0)
print(U*V)
