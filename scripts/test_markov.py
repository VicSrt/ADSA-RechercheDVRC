#!/usr/bin/env python
# coding: utf8

import numpy as np


def power_matrix(matrix,power):
    for i in range(power):
        matrix = matrix.dot(matrix)
    return matrix

def sum_vect(array):
    sum = 0
    for e in array:
        sum += e
    return sum

def compute_state_n(pi0,matrix,n):
    power_mat = power_matrix(matrix, n)
    pi = pi0.dot(power_mat)
    print("state"+str(n)+":"+str(pi))
    return pi


matrix = np.array((
            (0.25,0.25,0,0.25,0,0.25),
            (0.5,0,0,0.5,0,0),
            (0,0,1,0,0,0),
            (0,0,0,0.5,0.5,0),
            (0,0,0,0,1,0),
            (0.5,0,0,0,0,0.5),
         ))



def matrix_absorbtion(Q,R):
    M = np.identity(len(Q)) - Q
    N = np.linalg.inv(M)
    return  N.dot(R)

pi0 = np.array((1,0,0,0,0,0))

compute_state_n(pi0,matrix,0)
#compute_state_n(pi0,matrix,1)

#compute_state_n(pi0,matrix,10)
