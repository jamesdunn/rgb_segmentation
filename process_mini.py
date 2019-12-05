#!/usr/bin/python3

import math
import numpy as np
import os
import pdb
import sys

from scipy import stats

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
np.core.arrayprint._line_width = sys.maxsize

def main():
    project_name = sys.argv[1]
    image_width = float(sys.argv[2])
    image_height = float(sys.argv[3])
    scale_factor = float(sys.argv[4])

    scaled_width = math.floor(image_width / scale_factor)
    scaled_height = math.floor(image_height / scale_factor)

    # Load LAS txt file into numpy arrays.
    points = np.loadtxt('unclassified_cloud.txt', usecols=range(3))
    #classes = np.loadtxt('unclassified_cloud.txt', usecols=range(3,4), dtype=int)
    rgb = np.loadtxt('unclassified_cloud.txt', usecols=range(4,7), dtype=int)
    offset = np.loadtxt('offset.txt', usecols=range(3))
    p_matrices = np.loadtxt('pmatrix.txt', usecols=range(1,13))
    filenames = np.genfromtxt('pmatrix.txt', usecols=range(1), dtype=str)

    # Calculate the coordinates of projected 2D points, u and v.
    offset = np.tile(offset, (points.shape[0],1))
    prime = np.append(np.subtract(points, offset), np.tile(np.array([1]), (points.shape[0],1)), axis=1)
    

    # Initialize the array that hold the class values for each point.
    classes = np.zeros(shape=(points.shape[0],p_matrices.shape[0]))
    final_class = np.zeros(shape=(points.shape[0]))

    # For each image
    for i in range(p_matrices.shape[0]):
        
        filename = filenames[i]
        # Try opening the segmented image array.
        try:
            segmented_image = np.loadtxt(filename + '.txt', dtype=int)
            print(filename)
        except IOError as e:
            continue

        # For each image, calculate the point-to-pixel projection.
        p_matrix = p_matrices[i].reshape(3, 4)
        xyz = np.transpose(np.matmul(p_matrix, np.transpose(prime)))
        x = np.split(xyz, 3, 1)[0]
        y = np.split(xyz, 3, 1)[1]
        z = np.split(xyz, 3, 1)[2]
        u = np.true_divide(x, z)
        v = np.true_divide(y, z)
        coordinates = np.true_divide(np.append(u, v, axis=1), 3).astype(int)

        #Convert classifications from Unified Parsing to LAS format.
        # for row in segmented_image:
        #     for element in row:
        #         if element.astype(int) in [4, 97, 14, 37, 119]: # Vegetation
        #             element = 4
        #         elif element.astype(int) in [5, 335, 70, 9, 1]: # Building
        #             element = 6
        #         elif element.astype(int) in [2]: # Ground
        #             element = 22
        #         else: # Unclassified
        #             element = 0

        # For each point
        for j in range(points.shape[0]):
            # If the projected pixel is in the range of this image (i.e., it exists in this image).
            if (coordinates[j][0] in range(0,scaled_width) and coordinates[j][1] in range(0,scaled_height)):
                # Add pixel's class to list for point
                classes[j][i] = segmented_image[coordinates[j][1]][coordinates[j][0]]
                print('Point number ', j)   

    # Take most common classification amoung pixels correspoinding to this point as the point's class
    for k in range(classes.shape[0]):
        final_class[k] = stats.mode(classes[k])[0][0]
        # TODO: Use Bayesian filter to create confidence value, mapped to intensity in point cloud.

    # Merge everything together.
    classified_cloud = np.append(np.append(points, final_class.reshape(-1, 1).astype(int), axis=1), rgb.astype(int), axis=1)

    # Write the output file.
    output_file = open('classified_cloud.txt', 'w')
    print(np.array2string(classified_cloud).replace('\n','').replace(']','\n').replace('[',' ').replace('    ',' ').replace('   ',' ').replace('  ',' '), file = output_file)
    output_file.close()


if __name__ == "__main__":
    main()
