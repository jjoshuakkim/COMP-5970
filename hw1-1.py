import numpy as np
import pandas as pd
import scipy
from plotnine import *

'''
Homework 1 problem 8 -- global alignment
use the simple scoring method of +1 for match and -1 for mismatch and indel
print the global alignment with one line per string with '-' characters for indels
'''
def global_alignment(sequence1, sequence2):
    matchingMatrix = np.zeros((len(sequence1), len(sequence2)))     # Builds a matching patterns matrix initialized with 0s

    matrix = []                                                     # Builds matrix of 0s
    for i in range(len(sequence1)+1):
        subMatrix = []
        for j in range(len(sequence2)+1):
            subMatrix.append(0)
        matrix.append(subMatrix)
    
    for j in range(1, len(sequence2)+1):                            # Fills matrix using the alg
        matrix[0][j] = j*-1
    for i in range(1, len(sequence1)+1):
        matrix[i][0] = i*-1 
    
    match = 1
    mismatch = -1
    
    for i in range(len(sequence1)):                                 # Fills the matching matrix based on match or mismatch
        for j in range(len(sequence2)):
            if sequence1[i] == sequence2[j]:
                matchingMatrix[i][j] = match
            else:
                matchingMatrix[i][j] = mismatch

    for i in range(1, len(sequence1)+1):
        for j in range(1, len(sequence2)+1):
            matrix[i][j] = max(matrix[i-1][j-1]+matchingMatrix[i-1][j-1], 
                matrix[i-1][j]-1, matrix[i][j-1]-1)                 # Fills matrix based on highest values

    alignment1, alignment2, i, j = "", "", len(sequence1), len(sequence2)
    while (i > 0 and j > 0):
        if (i > 0 and j > 0 and matrix[i][j] == matrix[i-1][j-1] + matchingMatrix[i-1][j-1]):
            alignment1 = sequence1[i-1] + alignment1
            alignment2 = sequence2[j-1] + alignment2
            i = i - 1
            j = j - 1

        elif (i > 0 and matrix[i][j] == matrix[i-1][j] - 1):
            alignment1 = sequence1[i-1] + alignment1
            alignment2 = "-" + alignment2
            i = i - 1

        else:
            alignment1 = "-" + alignment1
            alignment2 = sequence2[j-1] + alignment2
            j = j - 1

    print(alignment1)
    print(alignment2)
'''
support code for creating random sequence, no need to edit
'''
def random_sequence(n):
    return("".join(np.random.choice(["A","C","G","T"], n)))

'''
support code for mutating a sequence, no need to edit
'''
def mutate(s, snp_rate, indel_rate):
    x = [c for c in s]
    i = 0
    while i < len(x):
        if np.random.random() < snp_rate:
            x[i] = random_sequence(1)
        if np.random.random() < indel_rate:
            length = np.random.geometric(0.5)
            if np.random.random() < 0.5: # insertion
                x[i] = x[i] + random_sequence(length)
            else:
                for j in range(i,i+length):
                    if j < len(x):
                        x[j] = ""
                    i += 1
        i += 1
    return("".join(x))

# creating related sequences
s1 = random_sequence(100)
s2 = mutate(s1, 0.1, 0.1)

# running your alignment code
global_alignment(s1, s2)
