import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt

DX = [1, -1, 0, 0]
DY = [0, 0, 1, -1]
SIZE_LOW_THRESHOLD = 0.02
SIZE_HIGH_THRESHOLD = 0.6
RATIO_THRESHOLD = 2.0
DENSITY_THRESHOLD = 0.4

def dfs(mask, md, x, y):
	mask[x][y] = 0
	md[0] = min(md[0], x)
	md[1] = min(md[1], y)
	md[2] = max(md[2], x)
	md[3] = max(md[3], y)
	md[4] += 1

	if md[4] > 900:
		return

	for i in range(0, 4):
		new_x = x + DX[i]
		new_y = y + DY[i]

		if new_x < 0 or new_x >= mask.shape[0] or new_y < 0 or new_y >= mask.shape[1] or mask[new_x][new_y] == 0:
			continue
		
		dfs(mask, md, new_x, new_y)

def get_region(im, mask, det, scale):
	im_size = im.shape[0] * im.shape[1]
	target_md = [mask.shape[0], 0, mask.shape[1], 0, 0]
	for i in range(0, mask.shape[0]):
		for j in range(0, mask.shape[1]):
			if mask[i][j] != 0:
				md = [mask.shape[0], mask.shape[1], 0, 0, 0]
				dfs(mask, md, i, j)
				pupil_size = (md[2] - md[0] + 1) * (md[3] - md[1] + 1)
				if float(pupil_size) / im_size < SIZE_LOW_THRESHOLD or float(pupil_size) / im_size > SIZE_HIGH_THRESHOLD:
					continue
				md.append(np.float32(md[4]) / pupil_size)
				if md[5] < DENSITY_THRESHOLD:
					continue
				ratio = np.float32(md[2] - md[0] + 1) / (md[3] - md[1] + 1)
				if ratio > RATIO_THRESHOLD or ratio < 1 / RATIO_THRESHOLD:
					continue
				if target_md[4] < md[4]:
					target_md = md

	if target_md[4] != 0:

		# if gaze == 'FORWARD':
		# 	org_im = org_im[:, width / 4 : width * 3 / 4]
		# 	target_md[1] += im.size()
		# elif gaze == 'LEFTWARD':
		# 	org_im = org_im[:, width / 2 : width]
		# elif gaze == 'RIGHTWARD':
		# 	org_im = org_im[:, 0 : width / 2]
		# else:
		# 	org_im = org_im[0 : height * 3 / 4, width / 4 : width * 3 / 4]

		return [
			np.uint16(np.around(det[0] + scale * target_md[1])),
			np.uint16(np.around(det[1] + scale * target_md[0])),
			np.uint16(np.around(det[0] + scale * target_md[3])),
			np.uint16(np.around(det[1] + scale * target_md[2])),
			target_md[5],
			'pupil'
		]

def get_regoin_from_circle(im, mask, det, scale, circle):
	circle_size = circles[2] ** 2 * 4
	target_md = [mask.shape[0], 0, mask.shape[1], 0, 0]
	for i in range(0, mask.shape[0]):
		for j in range(0, mask.shape[1]):
			if mask[i][j] != 0:
				print "dshfkjds"
				md = [mask.shape[0], mask.shape[1], 0, 0, 0]
				dfs(mask, md, i, j)
				pupil_size = (md[2] - md[0] + 1) * (md[3] - md[1] + 1)
				# if float(pupil_size) / circle_size < 0.2:
				# 	continue
				md.append(np.float32(md[4]) / pupil_size)
				# if md[5] < DENSITY_THRESHOLD:
				# 	continue
				ratio = np.float32(md[2] - md[0] + 1) / (md[3] - md[1] + 1)
				# if ratio > RATIO_THRESHOLD or ratio < 1 / RATIO_THRESHOLD:
				# 	continue
				if target_md[4] < md[4]:
					target_md = md

	if target_md[4] != 0:
		return [
			np.uint16(np.around(det[0] + scale * target_md[1])),
			np.uint16(np.around(det[1] + scale * target_md[0])),
			np.uint16(np.around(det[0] + scale * target_md[3])),
			np.uint16(np.around(det[1] + scale * target_md[2])),
			target_md[5],
			'pupil'
		]

def locate_pupil(org_im, det, gaze, black=False):
	im = org_im
	scale = 1
	max_side = 60
	if max(im.shape[:2]) > max_side:
		scale = np.float32(max(im.shape[:2])) / max_side
		im = cv2.resize(im, (np.int(im.shape[1] / scale), np.int(im.shape[0] / scale)))

	# height, width = im.shape[:2]
	# if gaze == 'FORWARD':
	# 	im = im[:, width / 4 : width * 3 / 4]
	# elif gaze == 'LEFTWARD':
	# 	im = im[:, width / 2 : width]
	# elif gaze == 'RIGHTWARD':
	# 	im = im[:, 0 : width / 2]
	# else:
	# 	im = im[0 : height * 3 / 4, width / 4 : width * 3 / 4]
	
	# cv2.imshow("images", im)
	# cv2.waitKey(0)

	hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	lower, upper = {}, {}

	color_set = [['red1', 'red2'], ['yellow1', 'yellow2']]
	if black:
		color_set.append(['black1', 'black2'])
	
	lower['null'] = np.array([1, 1, 1])
	upper['null'] = np.array([0, 0, 0])

	lower['red1'] = np.array([0, 160, 20])
	upper['red1'] = np.array([7, 255, 180])

	lower['red2'] = np.array([175, 140, 20])
	upper['red2'] = np.array([185, 255, 180])

	lower['yellow1'] = np.array([0, 160, 80])
	upper['yellow1'] = np.array([25, 230, 250])

	lower['yellow2'] = np.array([35, 70, 40])
	upper['yellow2'] = np.array([50, 100, 90])

	lower['black1'] = np.array([150, 0, 0])
	upper['black1'] = np.array([255, 255, 30])

	lower['black2'] = np.array([0, 0, 10])
	upper['black2'] = np.array([60, 255, 50])

	# lower['white'] = np.array([10, 20, 100])
	# upper['white'] = np.array([255, 60, 160])

	# lower['white2'] = np.array([10, 20, 100])
	# upper['white2'] = np.array([60, 60, 160])

	for colors in color_set:
		mask = cv2.inRange(hsv, lower['null'], upper['null'])
		for color in colors:
			mask += cv2.inRange(hsv, lower[color], upper[color])

		# out = cv2.bitwise_and(im, im, mask = mask)
		# cv2.imshow("images", np.hstack([im, out]))
		# cv2.waitKey(0)
		# cv2.imshow("images", mask)
		# cv2.waitKey(0)

		region = get_region(im, mask, det, scale)

		if region is not None:
			return region

	color_set.append(['white1', 'white2'])
	lower['white1'] = np.array([10, 20, 100])
	upper['white1'] = np.array([255, 60, 160])
	lower['white2'] = np.array([10, 20, 100])
	upper['white2'] = np.array([60, 60, 160])

	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	circles = cv2.HoughCircles(im_gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 15,
		param1=50, param2=20, minRadius=0, maxRadius=0)

	if circles is None:
		return
	
	circles = np.uint16(np.around(circles))
	for circle in circles[0,:]:
		for colors in color_set:
			mask = cv2.inRange(hsv, lower['null'], upper['null'])
			for color in colors:
				mask += cv2.inRange(hsv, lower[color], upper[color])
			for i in range(mask.shape[0]):
				for j in range(mask.shape[1]):
					if (i - circle[1]) ** 2 + (j - circle[0]) ** 2 > circle[2] ** 2:
						mask[i][j] = 0

			region = get_region(im, mask, det, scale)

			if region is not None:
				return region

	return [
		np.uint16(np.around(det[0] + scale * (circles[0][0][0] - circles[0][0][2]))),
		np.uint16(np.around(det[1] + scale * (circles[0][0][1] - circles[0][0][2]))),
		np.uint16(np.around(det[0] + scale * (circles[0][0][0] + circles[0][0][2]))),
		np.uint16(np.around(det[1] + scale * (circles[0][0][1] + circles[0][0][2]))),
		1,
		'pupil'
	]








	
