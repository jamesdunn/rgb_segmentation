import numpy as np
import pptk


def main():
	points = np.loadtxt('classified_cloud.txt', usecols=range(4))
	w = [0.8, 0.8, 0.8]
	r = [0.6, 0.3, 0.3]
	g = [0.3, 0.6, 0.3]
	b = [0.3, 0.3, 0.6]
	colors = np.array([w, r,g,b])
	classifications = np.unique(points[:,3])
	rgb = np.zeros((points.shape[0], 3))
	xyz = points[:,:3]
	for i in range(classifications.shape[0]):
		k = classifications[i]
		color = colors[i]
		rgb[np.where(points[:,3] == k)] = color
		a = points[np.where(points[:,3] == k)]
		print("Number of points in class %d: %d" % (k, a.shape[0]))
	pptk.viewer(xyz, rgb)


if __name__ == "__main__":
	main()