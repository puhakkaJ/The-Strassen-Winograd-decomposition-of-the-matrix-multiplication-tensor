
import numpy as np

# init

A = np.array([[0,1,0,0,0,0,1], [-1,-1,0,1,1,0,0], [0,-1,1,1,0,1,0], [1,1,-1,-1,0,0,0]])
B = np.array([[0,0,0,0,1,0,1], [0,1,1,1,1,0,0], [1,0,0,1,1,1,0], [1,0,1,1,1,0,0]])
C = np.array([[0,0,0,0,0,1,1], [0,0,1,-1,1,1,0], [1,1,0,1,0,-1,0], [1,0,-1,1,0,-1,0]])

U = np.array([[1,2],[3,4]])
V = np.array([[5,6],[7,8]])



# 1.a) M_2 tensor

M2 = np.array([[[0] * 4] * 4] * 4)
num = 0
n = 2

for i in range(0,2):
    for i2 in range(0,2):
        for j in range(0,2):
            for j2 in range(0,2):
                for k in range(0,2):
                    for k2 in range(0,2):
                        if (i==i2) and (j==j2) and (k==k2):
                            M2[i*n+k2][i2*n+j][j2*n+k] = 1
                        
print("$M_2 =$",M2,"\n")

# 1.b) Kruskalproduct tensor K(A, B, C)

T = np.array([[[0] * len(A)] * len(B)] * len(C))

for i in range(0,len(A)):
    for j in range(0,len(B)):
        for k in range(0,len(C)):
            sum = 0
            for l in range(0, len(A[0])):
                sum += A[i][l]*B[j][l]*C[k][l]
            
            T[i][j][k] = sum

print("K(A, B, C) =", T,"\n")

# 1.c) Product matrix W = U V

W = np.dot(U,V)

print("W =", W,"\n")

# 1.d) Vectorizations U_ , V_ , and W_

U_ = U.reshape(1,len(U)*len(U[0]))[0]
V_ = V.reshape(1,len(V)*len(V[0]))[0]
W_ = W.reshape(1,len(W)*len(W[0]))[0]

print("U_ =",U_,"\n")
print("V_ =",V_,"\n")
print("W_ =",W_,"\n")

# 1.e) vectors M_2[U_,V_] and A((B^T U_)⊙(C^T V_)).

a = []

for i in range(0,len(M2)):
    sum = 0
    for j in range(0,len(U_)):
        for k in range(0,len(V_)):
            sum += M2[i][j][k]*U_[j]*V_[k]           
    a.append(sum)

print("M_2[U_,V_] =", a)

I1 = np.dot(np.transpose(B),U_)
I2 = np.dot(np.transpose(C),V_)

I3 = [0] * len(I1)

for i in range(0, len(I1)):
    I3[i] = I1[i]*I2[i]

I4 = np.dot(A,I3)

print("A((B^T U_)⊙(C^T V_)) =", I4)


