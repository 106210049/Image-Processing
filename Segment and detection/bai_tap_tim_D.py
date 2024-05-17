import numpy as np
import math
M=1500
N=1000
D=10
X1=np.random.randn(M,D)
X2=np.random.randn(N,D)
H=np.zeros((M,N))
for i in range(0,M):
    for j in range(0,N):
        distance=np.linalg.norm(X1[i]-X2[j])
        H[i,j]=distance

print("Ma tran H: ")
print(H)
print("Kich thuoc ma tran H")
print(H.shape)

X12=np.sum(X1**2,axis=1,keepdims=True)
X22=np.sum(X2**2,axis=1,keepdims=True)
K=np.sqrt(X12+X22.T-2*np.dot(X1,X2.T))
print("ma tran moi")
print(K)
print(K.shape)