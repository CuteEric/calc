# !usr/bin/env
# coding=utf-8
from glob import glob
import cv2
import numpy as np
import os
from random import Random
import math
import time
from sys import platform

def drawMatches(img1, kp1, img2, kp2):
	# Create a new output image that concatenates the two images together
	rows1 = img1.shape[0]
	cols1 = img1.shape[1]
	rows2 = img2.shape[0]
	cols2 = img2.shape[1]

	out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

	# Place the first image to the left
	out[:rows1, :cols1] = np.dstack([img1, img1, img1])

	# Place the next image to the right of it
	out[:rows2, cols1:] = np.dstack([img2, img2, img2])

	# For each pair of points we have between both images
	# draw circles, then connect a line between them
	for i in range(np.size(kp1, 0)):
		# Get the matching keypoints for each of the images
		# x - columns
		# y - rows

		x1 = kp1[i, 0]
		y1 = kp1[i, 1]
		x2 = kp2[i, 0]
		y2 = kp2[i, 1]
		
		# Draw a small circle at both co-ordinates
		# radius 4
		# colour blue
		# thickness = 1
		cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)   
		cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

		# Draw a line in between the two points
		# thickness = 1
		# colour blue
		cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)


	# Show the image
	cv2.imshow('Matched Features', out)
	cv2.waitKey(0)

	# Also return the image if you'd like a copy
	return out

def randPerspectiveWarp(im, w, h, r, ret_pts=False):

	"""
	Applies a pseudo-random perspective warp to an image.
	input:
	im - the original image
	h - image height
	w - image width
	r - Random instance
	returns:
	im_warp - the warped image
	ret_pts - if True, return the points generated 
	""" 

	# Generate two pseudo random planes within tolerances for the projective transformation of the original image
	# Each point is from the center half of its respective x y quandrant. openCV getPerpectiveTransform expects [Q2, Q3, Q1, Q4] for points in each image quandrant, so that it 
	# the iteration order here. Note that 0,0 is the top left corner of the picture. Additionally, we can only perform a tranformation to zoom in, since exprapolated pixels
	# look unnatural, and will ruin the similarity between the two images.

	# limits for random number generation
	minsx = [ 0, 3*w/4 ]
	maxsx = [ w/4, w ]
	minsy = [ 0, 3*h/4 ]
	maxsy = [ h/4, h ]


	pts_orig = np.zeros((4, 2), dtype=np.float32) # four original points
	pts_warp = np.zeros((4, 2), dtype=np.float32) # points for the affine transformation. 

	# fixed point for the first plane	
	pts_orig[0, 0] = 0
	pts_orig[0, 1] = 0
	
	pts_orig[1, 0] = 0
	pts_orig[1, 1] = h

	pts_orig[2, 0] = w
	pts_orig[2, 1] = 0

	pts_orig[3, 0] = w
	pts_orig[3, 1] = h

	# random second plane
	pts_warp[0, 0] = r.uniform(minsx[0], maxsx[0])#用于生成一个指定范围内的随机浮点数，两格参数中，其中一个是上限，一个是下限
	pts_warp[0, 1] = r.uniform(minsy[0], maxsy[0])
	
	pts_warp[1, 0] = r.uniform(minsx[0], maxsx[0])
	pts_warp[1, 1] = r.uniform(minsy[1], maxsy[1])

	pts_warp[2, 0] = r.uniform(minsx[1], maxsx[1])
	pts_warp[2, 1] = r.uniform(minsy[0], maxsy[0])

	pts_warp[3, 0] = r.uniform(minsx[1], maxsx[1])
	pts_warp[3, 1] = r.uniform(minsy[1], maxsy[1])

	# compute the 3x3 transform matrix based on the two planes of interest
	T = cv2.getPerspectiveTransform(pts_warp, pts_orig)#获得透视变换矩阵

	# apply the perspective transormation to the image, causing an automated change in viewpoint for the net's dual input
	im_warp = cv2.warpPerspective(im, T, (w, h))#进行透视变换矩阵
	if not ret_pts:
		return im_warp
	else: 
		return im_warp, pts_warp

def showImWarpEx(im_fl, save):
	"""
	Show an example of warped images and their corresponding four corner points.
	"""

	im = cv2.resize(cv2.cvtColor(cv2.imread(im_fl), cv2.COLOR_BGR2GRAY), (256, int(120./160 * 256)))
	r = Random(0)
	r.seed(time.time())
	h, w = im.shape
	im_warp, pts_warp = randPerspectiveWarp(im, w, h, r, ret_pts=True) # get the perspective warped picture	
	
	pts_orig = np.zeros((4, 2), dtype=np.float32) # four original points
	ofst = 3
	pts_orig[0, 0] = ofst
	pts_orig[0, 1] = ofst	
	pts_orig[1, 0] = ofst
	pts_orig[1, 1] = h-ofst
	pts_orig[2, 0] = w-ofst
	pts_orig[2, 1] = ofst
	pts_orig[3, 0] = w-ofst
	pts_orig[3, 1] = h-ofst

	pts_rect = np.zeros((4, 2), dtype=np.float32) # for creating rectangles
	pts_rect[0, 0] = w/4
	pts_rect[0, 1] = h/4	
	pts_rect[1, 0] = w/4
	pts_rect[1, 1] = 3*h/4
	pts_rect[2, 0] = 3*w/4
	pts_rect[2, 1] = h/4
	pts_rect[3, 0] = 3*w/4
	pts_rect[3, 1] = 3*h/4

	# save orig before placing rectangles on it
	if save:
		cv2.imwrite('Original.jpg', im)
	
	for i in range(4):
		cv2.rectangle(im, (pts_orig[i,0], pts_orig[i,1]), (pts_rect[i,0], pts_rect[i,1]), (255, 255, 255), thickness=2)
	
	print pts_orig[:]
	out_im = drawMatches(im, pts_warp, im_warp, pts_orig)
	if save:
		cv2.imwrite("Warped.jpg", im_warp)
		print 'Images saved in current directory'

if __name__ == '__main__':
	img_file = './000000.png'
	showImWarpEx(img_file, True)