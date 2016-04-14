
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2


imageA = cv2.imread("Panorama\\img4.jpg")
imageB = cv2.imread("Panorama\\img2.jpg")
imageA = imutils.resize(imageA, width=200)
imageB = imutils.resize(imageB, width=200)

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imwrite("keypoint matches4.jpg" , vis)
cv2.imshow("Result", result)
cv2.imwrite("result4.jpg" , result)
cv2.waitKey(0)