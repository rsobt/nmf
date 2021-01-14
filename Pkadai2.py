from numpy import *
from nmf import *
A=matrix([[1,1,2,1,3],[2,3,3,4,4],[1,1,2,1,3],[1,2,1,3,1]])

for i in range(11,20):
    print("rand seed : " + str(i))
    U,V=factorize_S(A,dim=2,min_imp=10**(-30),seed=i)
    print(squared_frobenius_norm(A,U*V))
