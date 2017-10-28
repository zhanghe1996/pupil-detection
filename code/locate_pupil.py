import os
import cv2
import numpy as np

DX = [1, -1, 0, 0]
DY = [0, 0, 1, -1]
SIZE_THRESHOLD = 50
DENSITY_THRESHOLD = 0.4

def dfs(mask, md, x, y):
	mask[x][y] = 0
	md[0] = min(md[0], x)
	md[1] = min(md[1], y)
	md[2] = max(md[2], x)
	md[3] = max(md[3], y)
	md[4] += 1

	for i in range(0, 4):
		new_x = x + DX[i]
		new_y = y + DY[i]

		if new_x < 0 or new_x >= mask.shape[0] or new_y < 0 or new_y >= mask.shape[1] or mask[new_x][new_y] == 0:
			continue
		
		dfs(mask, md, new_x, new_y)

def get_region(im, mask, det, scale):
	target_md = [mask.shape[0], 0, mask[1], 0, 0]
	for i in range(0, mask.shape[0]):
		for j in range(0, mask.shape[1]):
			if mask[i][j] != 0:
				md = [mask.shape[0], mask.shape[1], 0, 0, 0]
				dfs(mask, md, i, j)
				if md[4] < SIZE_THRESHOLD:
					continue
				md.append(np.float32(md[4]) / ((md[2] - md[0] + 1) * (md[3] - md[1] + 1)))
				if md[5] < DENSITY_THRESHOLD:
					continue
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

def locate_pupil(org_im, det, black=False):
	im = org_im
	scale = 1
	if max(im.shape[:2]) > 100:
		scale = np.float32(max(im.shape[:2])) / 100
		im = cv2.resize(im, (np.int(im.shape[1] / scale), np.int(im.shape[0] / scale)))

	hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	lower, upper = {}, {}

	color_set = [['red1', 'red2'], ['yellow']]
	if black:
		color_set.append(['black1', 'black2'])
	
	lower['null'] = np.array([1, 1, 1])
	upper['null'] = np.array([0, 0, 0])

	lower['red1'] = np.array([0, 160, 20])
	upper['red1'] = np.array([7, 255, 180])

	lower['red2'] = np.array([170, 110, 20])
	upper['red2'] = np.array([185, 255, 180])

	lower['yellow'] = np.array([0, 160, 80])
	upper['yellow'] = np.array([25, 230, 250])

	lower['black1'] = np.array([90, 0, 20])
	upper['black1'] = np.array([130, 150, 55])

	lower['black2'] = np.array([0, 0, 10])
	upper['black2'] = np.array([60, 150, 40])

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

	
