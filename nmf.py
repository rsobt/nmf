#coding:utf-8
from numpy import *
import math

# 同じ形の行列AとBに対しA-Bのフロベニウスノルムの２乗を返す
def squared_frobenius_norm(A,B):
    norm=0
    for i in range(shape(A)[0]):
        for j in range(shape(A)[1]):
            norm+=pow(A[i,j]-B[i,j],2)
    return norm

def KL_divergence(A,B):
    divergence=0
    for i in range(shape(A)[0]):
        for j in range(shape(A)[1]):
            divergence+=A[i,j]*math.log(A[i,j]/B[i,j])-A[i,j]+B[i,j]
    return divergence


def IS_divergence(A,B):
    divergence=0
    for i in range(shape(A)[0]):
        for j in range(shape(A)[1]):
            divergence+=A[i,j]/B[i,j] - math.log(A[i,j]/B[i,j])-1
    return divergence

# Vの各行ベクトルの成分の値の和が１になるように正規化する．
# ただしUVの値は変わらないようにする．
def normalize_V(U,V):
    for i in range(shape(V)[0]):
        total=0
        for j in range(shape(V)[1]):
            total+=V[i,j]
        for j in range(shape(V)[1]):
            V[i,j]/=total
        for k in range(shape(U)[0]):
            U[k,i]*=total
    return U,V

# 非負値行列Aに対する非負値行列因子分解でえられる２つの行列U,Vを返す
def factorize(A,dim=10,iteration_num=100,seed=0):
    # Aの行数と列数を得る
    dim_row,dim_column=shape(A)

    # ランダムに値を埋めた初期行列U,Vを求める
    random.seed(seed)
    U=matrix([[random.random() for j in range(dim)] for i in range(dim_row)])
    V=matrix([[random.random() for j in range(dim_column)] for i in range(dim)])

    #更新式を指定した回数だけ繰り返し適用する
    for i in range(iteration_num):
        # 現在のA-UVのフロベニウスノルムの２乗の値costを計算する
        cost=squared_frobenius_norm(A,U*V)
        # 10回毎にcostを表示する
        #if i%10==0: print(cost)
        # すでにA=UVとなる行列分解を達成していれば抜ける
        if cost==0: break


        # Uの更新を行う
        UU=matrix([[double(0) for j in range(dim)] for i in range(dim_row)])
        for i in range(dim_row):
            for k in range(dim):
                UU[i,k] = U[i,k] * (A*V.T)[i,k] / (U*V*V.T)[i,k]
        U = UU
        
        # Vの更新を行う
        VV=matrix([[double(0) for j in range(dim_column)] for i in range(dim)])
        for k in range(dim):
            for j in range(dim_column):
                VV[k,j] = V[k,j] * (A.T*U)[j,k] / (U.T*U*V)[k,j]
        V = VV

    # UVの値を変えずにVのベクトルを正規化してからU,Vを返す
    return normalize_V(U,V)

# 非負値行列Aに対する非負値行列因子分解でえられる２つの行列U,Vを返す
#Ended loop by min_imp
def factorize_S(A,dim=10,min_imp=10**(-20),seed=0):
    # Aの行数と列数を得る
    dim_row,dim_column=shape(A)

    # ランダムに値を埋めた初期行列U,Vを求める
    random.seed(seed)
    U=matrix([[random.random() for j in range(dim)] for i in range(dim_row)])
    V=matrix([[random.random() for j in range(dim_column)] for i in range(dim)])

    #更新式を指定した回数だけ繰り返し適用する
    tmp = 0
    while 1:
        tmp += 1
        # 現在のA-UVのフロベニウスノルムの２乗の値costを計算する
        cost=squared_frobenius_norm(A,U*V)
        # 10回毎にcostを表示する
        #if i%10==0: print(cost)
        # すでにA=UVとなる行列分解を達成していれば抜ける
        if cost==0: break


        # Uの更新を行う
        UU=matrix([[double(0) for j in range(dim)] for i in range(dim_row)])
        for i in range(dim_row):
            for k in range(dim):
                UU[i,k] = U[i,k] * (A*V.T)[i,k] / (U*V*V.T)[i,k]
        U = UU
        
        # Vの更新を行う
        VV=matrix([[double(0) for j in range(dim_column)] for i in range(dim)])
        for k in range(dim):
            for j in range(dim_column):
                VV[k,j] = V[k,j] * (A.T*U)[j,k] / (U.T*U*V)[k,j]
        V = VV

        if abs(cost-squared_frobenius_norm(A,U*V))<min_imp:
            break

    # UVの値を変えずにVのベクトルを正規化してからU,Vを返す
    return normalize_V(U,V)

def factorize_KL(A,dim=10,iteration_num=100,seed=0):
    # Aの行数と列数を得る
    dim_row,dim_column=shape(A)

    # ランダムに値を埋めた初期行列U,Vを求める
    random.seed(seed)
    U=matrix([[random.random() for j in range(dim)] for i in range(dim_row)])
    V=matrix([[random.random() for j in range(dim_column)] for i in range(dim)])

    #更新式を指定した回数だけ繰り返し適用する
    for i in range(iteration_num):
        #caluculate cost
        cost=KL_divergence(A,U*V)
        # 10回毎にcostを表示する
        #if i%10==0: print(cost)
        # すでにA=UVとなる行列分解を達成していれば抜ける
        if cost==0: break


        # Uの更新を行う
        UU=matrix([[double(0) for j in range(dim)] for i in range(dim_row)])
        UV = U*V
        TMP = matrix([double(1) for i in range(dim_column)])
        for i in range(dim_row):
            for k in range(dim):
                tmp = 0
                for j in range(dim_column):
                    tmp += A[i,j]*V[k,j]/UV[i,j]
                UU[i,k] = U[i,k] * tmp / (V[k,:]*TMP.T)
        U = UU
        
        # Vの更新を行う
        VV=matrix([[double(0) for j in range(dim_column)] for i in range(dim)])
        UV = U*V
        TMP = matrix([double(1) for i in range(dim_row)])
        for k in range(dim):
            for j in range(dim_column):
                tmp = 0
                for i in range(dim_row):
                    tmp += A[i,j]*U[i,k]/UV[i,j]
                VV[k,j] = V[k,j] * tmp / (U[:,k].T*TMP.T)
        V = VV

    # UVの値を変えずにVのベクトルを正規化してからU,Vを返す
    return normalize_V(U,V)

def factorize_IS(A,dim=10,iteration_num=100,seed=0):
    # Aの行数と列数を得る
    dim_row,dim_column=shape(A)

    # ランダムに値を埋めた初期行列U,Vを求める
    random.seed(seed)
    U=matrix([[random.random() for j in range(dim)] for i in range(dim_row)])
    V=matrix([[random.random() for j in range(dim_column)] for i in range(dim)])

    #更新式を指定した回数だけ繰り返し適用する
    for i in range(iteration_num):
        #caluculate cost
        cost=KL_divergence(A,U*V)
        # 10回毎にcostを表示する
        #if i%10==0: print(cost)
        # すでにA=UVとなる行列分解を達成していれば抜ける
        if cost==0: break


        # Uの更新を行う
        UU=matrix([[double(0) for j in range(dim)] for i in range(dim_row)])
        UV = U*V
        TMP = matrix([double(1) for i in range(dim_column)])
        for i in range(dim_row):
            for k in range(dim):
                tmp1 = 0
                tmp2 = 0
                for j in range(dim_column):
                    tmp1 += A[i,j]*V[k,j]/UV[i,j]**2
                    tmp2 += V[k,j]/UV[i,j]
                UU[i,k] = U[i,k] * sqrt ( tmp1 / tmp2 )
        U = UU
        
        # Vの更新を行う
        VV=matrix([[double(0) for j in range(dim_column)] for i in range(dim)])
        UV = U*V
        TMP = matrix([double(1) for i in range(dim_row)])
        for k in range(dim):
            for j in range(dim_column):
                tmp1 = 0
                tmp2 = 0
                for i in range(dim_row):
                    tmp1 += A[i,j]*U[i,k]/UV[i,j]**2
                    tmp2 += U[i,k]/UV[i,j]
                VV[k,j] = V[k,j] * sqrt( tmp1 / tmp2 )
        V = VV

    # UVの値を変えずにVのベクトルを正規化してからU,Vを返す
    return normalize_V(U,V)
