#!/usr/bin/python3

import math
import numpy as np
import os
import pdb
import sys
from scipy import stats
from glob import glob

from occlusion_detection import get_occlusion_mask, get_occlusion_mask2

# Some hacky print options to get numpy to print in the format that txt2las wants.
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
np.core.arrayprint._line_width = sys.maxsize

def main():

    image_width = float(sys.argv[1])
    image_height = float(sys.argv[2])
    scale_factor = float(sys.argv[3])

    scaled_width = math.floor(image_width / scale_factor)
    scaled_height = math.floor(image_height / scale_factor)

    # Load values from the LAS txt file into numpy arrays.
    points = np.loadtxt('unclassified_cloud.txt', usecols=range(3))
    rgb = np.loadtxt('unclassified_cloud.txt', usecols=range(4,7), dtype=int)
    offset = np.loadtxt('offset.txt', usecols=range(3))
    p_matrices = np.loadtxt('pmatrix.txt', usecols=range(1,13))
    filenames = np.genfromtxt('pmatrix.txt', usecols=range(1), dtype=str)

    print('Loaded LAS file.')

    # Account for the image offset provided by Pix4D.
    offset = np.tile(offset, (points.shape[0],1))
    prime = np.hstack((points - offset, np.ones((points.shape[0], 1))))

    # Initialize the array that holds the class values for each point.
    classes = np.zeros(shape=(points.shape[0],p_matrices.shape[0]))
    final_class = np.zeros(shape=(points.shape[0]))

    # For each segmented image (given in txt format of 2D array with class labels).

    ITR_STEP = 5
    for i in range(0, p_matrices.shape[0], ITR_STEP):
        filename = filenames[i]
        # Try opening. Skip if it isn't present.
        try:
            segmented_image = np.loadtxt('unifiedparsing/segmented_images/' + filename + '.txt', dtype=int)
            print(filename + ' found.')
        except IOError as e:
            print(filename + ' not found.')
            continue

        # Calculate the point (x, y, z) to pixel (u, v) projection using Pix4D's pmatrix.
        p_matrix = p_matrices[i].reshape(3, 4)

        print("Obtaining Occlusion Mask...")
        occlusion_mask = get_occlusion_mask(prime, p_matrix, image_width, image_height)
        unoccluded_points = prime[occlusion_mask != 0]
        print("Total number of unoccluded points: %d" % unoccluded_points.shape[0])

        xyz = np.transpose(np.matmul(p_matrix, np.transpose(prime)))
        x = np.split(xyz, 3, 1)[0]
        y = np.split(xyz, 3, 1)[1]
        z = np.split(xyz, 3, 1)[2]
        u = np.true_divide(x, z)
        v = np.true_divide(y, z)
        coordinates = np.true_divide(np.append(u, v, axis=1), scale_factor).astype(int)

        # Convert classifications from Unified Parsing to LAS format.
        segmented_image[np.logical_not(np.isin(segmented_image, [4, 97, 14, 37, 119, 5, 335, 70, 9, 1, 22]))] = 0
        segmented_image[np.isin(segmented_image, [4, 97, 14, 37, 119])] = 4
        segmented_image[np.isin(segmented_image, [5, 335, 70, 9, 1])] = 6
        segmented_image[np.isin(segmented_image, [22])] = 2

        #Note: Computed valid coordinates (i.e. in camera frame) as part of the occlusion mask
        coordinates_mask = np.nonzero(occlusion_mask)
        valid_coordinates = coordinates[coordinates_mask]

        # Walk through each point, grab the classification for its corresponding 2D image coordinate.
        for j in range(valid_coordinates.shape[0]):
            #Index the coordinate mask to determine which elements in the class array to update
            index = coordinates_mask[0][j]
            classes[index][i] = segmented_image[valid_coordinates[j][1]][valid_coordinates[j][0]]

    # Take most common classification among pixels corresponding to this point as the point's final classification value.
    for k in range(classes.shape[0]):
        counts = np.bincount(classes[k].astype(int))
        counts[0] = 0
        final_class[k] = np.argmax(counts)
        # TODO: Use Bayesian filter to create confidence value, mapped to intensity in point cloud.
    print('Finished calculating final class for each point.')

    # Merge the point coordinates, classes, and RGB values together.
    classified_cloud = np.append(np.append(points, final_class.reshape(-1, 1).astype(int), axis=1), rgb.astype(int), axis=1)

    # Write the output txt file for txt2las.
    np.savetxt('classified_cloud.txt', classified_cloud, fmt="%f", delimiter=" ")

if __name__ == "__main__":
    main()
