import numpy as np
import cv2
import codecs

import os, sys

argCnt = len(sys.argv)

if argCnt != 4:
	print("Not enough arguments")
	print("Example: undistort.py /path/to/image /path/to/camera/matrix/file /path/to/distortion/file")
	sys.exit()


imagePath = sys.argv[1]
camMatPath = sys.argv[2]
distPath = sys.argv[3]


if not os.path.exists(imagePath):
  print("Invalid image:", imagePath)
  sys.exit()

if not os.path.exists(camMatPath):
  print("Invalid camera matrix:", camMatPath)
  sys.exit()

if not os.path.exists(distPath):
  print("Invalid distortion:", distPath)
  sys.exit()

print("Image Path:", imagePath)
print("Camera Matrix Path:", camMatPath)
print("Distortion Path:", distPath)

img = cv2.imread(imagePath)
print(distPath)

dist = np.loadtxt(distPath)
camMat = np.loadtxt(camMatPath)

height, width, channels = img.shape
map1, map2 = cv2.initUndistortRectifyMap(camMat, dist, np.eye(3), camMat, (width,height), cv2.CV_16SC2)
newImg = cv2.remap( img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imshow("new", newImg)
cv2.imshow("old", img)
cv2.waitKey()

