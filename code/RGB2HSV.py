import cv2
import numpy as np

if __name__ == '__main__':
	while True:
		r, g, b = raw_input("RGB: ").split()
		c = np.uint8([[[b, g, r]]])
		print cv2.cvtColor(c, cv2.COLOR_BGR2HSV)