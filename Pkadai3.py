from numpy import *
from nmf import *
A=matrix([[1,1,2,1,3],[2,3,3,4,4],[1,1,2,1,3],[1,2,1,3,1]])
for i in range(1,5):
    print("iteration_num : " + str(10**i))
    U,V=factorize_IS(A,dim=2,iteration_num=10**i,seed=0)
    print(KL_divergence(A,U*V))
    print(U*V)

