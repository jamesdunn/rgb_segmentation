#!/usr/bin/python3

import math
import numpy as np
import os
from scipy import stats
import sys

def main():

    points = np.loadtxt('unclassified_cloud.txt', usecols=range(3))
    classes = np.loadtxt('unclassified_cloud.txt', usecols=range(3,4))
    offset = np.loadtxt('offset.txt', usecols=range(3))
    p_matrices = np.loadtxt('pmatrix.txt', usecols=range(1,13))
    filenames = np.genfromtxt('pmatrix.txt', usecols=range(1), dtype=str)

    offset = np.tile(offset, (points.shape[0],1))

    prime = np.subtract(points, offset)

    extended = np.append(prime, np.tile(np.array([1]), (points.shape[0],1)), axis=1)

    p_matrix=p_matrices[29].reshape(3, 4)

    multiplied = np.matmul(p_matrix, np.transpose(extended))

    multrans = np.transpose(multiplied)

    x = np.split(multrans, 3, 1)[0]
    y = np.split(multrans, 3, 1)[1]
    z = np.split(multrans, 3, 1)[2]

    u = np.true_divide(x, z)
    v = np.true_divide(y, z)

    coordinates = np.true_divide(np.append(u, v, axis=1), 3).astype(int)

    segmented_image = np.loadtxt('DJI_0556.JPG.txt')

    for i in range(2651412):
        print(i)
        if (coordinates[i][0] in range(0,1333) and coordinates[i][1] in range(0,1000)):
            print('True')
            classes[i] = segmented_image[coordinates[i][1]][coordinates[i][0]]

    print(classes)

if __name__ == "__main__":
    main()
