import cv2
import numpy as np
from matplotlib import pyplot as plt
def showImage(in_img, in_path, out_img_vec, out_path):
	in1 = cv2.resize(cv2.imread(in_path + in_img), (300, 500))
	bar = np.zeros((in1.shape[0],5,3),np.uint8)
	add2cart = cv2.imread("W:\\Contests\\Myntra Hacakthon\\add2cart.JPG")
	count = 0

	while(1):
		cv2.imshow('in1',in1)
		k = 0xFF & cv2.waitKey(1)
		if k == ord('n'):
			out1 = cv2.resize(cv2.imread(out_path + out_img_vec[count%len(out_img_vec)]), (300, 500))
			res1 = np.hstack((in1 ,bar,out1))
			count += 1
			cv2.imshow('Search Result',res1)
		elif k == 27:
			cv2.destroyAllWindows()
			break