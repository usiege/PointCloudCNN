#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:2019/4/9 上午10:21
# software:PyCharm

import numpy as np

A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float
)

X = np.matrix([
    [i, -i] for i in range(A.shape[0])
], dtype=float)

print("A:"); print (A)
print("X:"); print (X)

print("A*X:"); print(A * X)

# self-loop
I = np.matrix(np.eye(A.shape[0]))
print("I:"); print (I)

A_hat = A + I
print('A_hat:(with self-loop)');
print(A_hat)
print(A_hat * X)

D = np.array(np.sum(A, axis=0))[0]
D = np.matrix(np.diag(D))

print("D:"); print(D)






