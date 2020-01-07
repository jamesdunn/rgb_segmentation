import numpy as np
from math import floor
import scipy.linalg as lin
from collections import defaultdict
import time
import skimage.measure

#Radius within which to check pixels for occlusion (base value), scales with distance from camera
#Note that for simplicity, this is treated as a window instead of an actual circle
#TODO: Compute this from the density of the point cloud
BASE_RAD = 10

#Fall-off adjustment factor for the search radius. Essentially, this just scales the radius fall-off so that
# at a distance of SCALE_FACTOR, the radius that points can within and not be occluded is BASE_RAD
SCALE_FACTOR = 20.0

#Compute the occlusion mask using block reduce - faster implementation, though might not work on a very spread out point cloud
#If neccessary, this can be modified to include several block reduce operations of different block sizes, and then for each point choosing
#the approriate one
def get_occlusion_mask(points, pmatrix, width, height):
	width = int(width)
	height = int(height)

	R = pmatrix[:, :3]
	t = pmatrix[:, 3]

	#Compute the location of the camera in world coordinates
	camera_loc = np.append(-np.dot(lin.pinv(R), t), 1)
	
	#Projecting the points, and rounding to the nearest pixel
	projected_points = np.transpose(np.matmul(pmatrix, np.transpose(points)))
	z = np.split(projected_points, 3, 1)[2].flatten()
	projected_points = projected_points / z[:, None]
	projected_points = np.rint(projected_points).astype(int)
	
	#Project the points, and filter out points that project outside of the camera image
	x_mask = np.isin(np.transpose(projected_points)[0], range(0,width ))
	y_mask = np.isin(np.transpose(projected_points)[1], range(0,height))
	coordinates_mask = np.logical_and(x_mask, y_mask)
	projected_points = projected_points[coordinates_mask]

	#Compute the projection distance of the points
	projected_distance = np.linalg.norm(points - camera_loc, axis=1)[coordinates_mask]

	#Construct a 2D array of the same size of the camera image
	#This will contain the minimum projection distances of the points that project to that pixel
	p_dist_image = np.full((width, height), np.inf)

	#Occlusion Mask for all points, initialized so that all points are considered occluded
	occlusion_mask = np.zeros((points.shape[0], 1))
	
	#Compute the minimum projection distances
	for i in range(projected_points.shape[0]):
		p_dist_image[tuple(projected_points[i,:2])] = min(p_dist_image[tuple(projected_points[i,:2])], projected_distance[i])

	#Apply the block reduce, output size will be a 2D array of (width / BASE_RAD, height / BASE_RAD)
	p_dist_image = skimage.measure.block_reduce(p_dist_image, (BASE_RAD,BASE_RAD), np.min)

	#Construct an occlusion mask of all the points that project into the camera image
	partial_occlusion_mask = np.zeros((projected_points.shape[0], 1))

	#Check for occlusion for each of the projected points
	for i in range(projected_points.shape[0]):
		#Compute the projection distance of a point with the minimum projection distance of the points at that location in the
		#block reduced array 
		comp_dist = projected_distance[i] - p_dist_image[tuple(np.floor_divide(projected_points[i,:2], BASE_RAD))]

		#Check if the comparison distance lies within a certain radius (in the direction of the camera projection) 
		partial_occlusion_mask[i] = 0.0 if comp_dist > BASE_RAD * (SCALE_FACTOR / projected_distance[i]) else 1.0

	occlusion_mask[coordinates_mask] = partial_occlusion_mask

	occlusion_mask = occlusion_mask.reshape((-1))
	return occlusion_mask

#This is mostly the same as the function above, but manually checks within the neighborhood (of differing sizes) of a pixel
#This is much slower than block-reducing 
def get_occlusion_mask_slow(points, pmatrix, width, height):
	width = int(width)
	height = int(height)

	start_time = time.time()
	#Location of the camera in world coordinates
	R = pmatrix[:, :3]
	t = pmatrix[:, 3]
	camera_loc = np.append(-np.dot(lin.pinv(R), t), 1)
	
	#Projecting the points, and rounding to the nearest pixel
	projected_points = np.transpose(np.matmul(pmatrix, np.transpose(points)))
	z = np.split(projected_points, 3, 1)[2].flatten()
	projected_points = projected_points / z[:, None]
	projected_points = np.rint(projected_points)

	x_mask = np.isin(np.transpose(projected_points)[0], range(0,width ))
	y_mask = np.isin(np.transpose(projected_points)[1], range(0,height))
	coordinates_mask = np.logical_and(x_mask, y_mask)
	projected_points = projected_points[coordinates_mask]

	projected_distance = np.linalg.norm(points - camera_loc, axis=1)[coordinates_mask]

	occlusion_mask = np.zeros((points.shape[0], 1))
	d = dict()
	for i in range(projected_points.shape[0]):
		if tuple(projected_points[i,:2]) in d.keys():
			d[tuple(projected_points[i,:2])] = min(projected_distance[i], d[tuple(projected_points[i,:2])])
		else:
			d[tuple(projected_points[i,:2])] = projected_distance[i]

	partial_occlusion_mask = np.zeros((projected_points.shape[0], 1))
	for i in range(projected_points.shape[0]):
		pixel_check_radius = floor(min(BASE_RAD * (SCALE_FACTOR / projected_distance[i]) , BASE_RAD))
		point_loc = projected_points[i,:2]

		unoccluded = np.zeros((2 * pixel_check_radius + 1, 2 * pixel_check_radius + 1))
		for h in range(-pixel_check_radius, pixel_check_radius + 1):
			for w in range(-pixel_check_radius, pixel_check_radius + 1):
				comp_dist = projected_distance[i] - d[tuple(point_loc + np.array([h, w]))] if tuple(point_loc + np.array([h, w])) in d.keys() else 0.0
				unoccluded[h, w] = 0.0 if comp_dist > BASE_RAD * (SCALE_FACTOR / projected_distance[i]) else 1.0
		partial_occlusion_mask[i] = unoccluded.all()

	occlusion_mask[coordinates_mask] = partial_occlusion_mask
	occlusion_mask = occlusion_mask.reshape((-1))
	total_time = time.time() - start_time
	print("Total time: %f" % total_time)

	return occlusion_mask






	