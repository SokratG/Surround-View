import numpy as np
import cv2
import codecs

import os, sys

argCnt = len(sys.argv)

if argCnt != 2:
	print("Not enough arguments")
	print("Example: undistort.py  -camera number")
	sys.exit()

camnum = int(sys.argv[1])
camMatPath = "video" + sys.argv[1]

resolutionX = 1920
resolutionY = 1080

dist = np.loadtxt(camMatPath + ".dist")
camMat = np.loadtxt(camMatPath + ".K")

video_device = [0, 1, 2, 3]
devName = "video" + str(video_device[camnum])
gSource = "v4l2src"
gDevice = " device=/dev/" + devName
gPipline = " ! video/x-raw,width={:d},height={:d},format=(string)UYVY ! nvvidconv ! video/x-raw(memory:NVMM),width={:d},height={:d},format=(string)I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink".format(resolutionX, resolutionY, resolutionX, resolutionY)

gCapture = gSource + gDevice + gPipline

cap = cv2.VideoCapture(gCapture)

while(cap.isOpened()):
	ret, img = cap.read() #cv.imread(fname)
	height, width, channels = img.shape
	shape = (width, height)
	newcammtx, roi = cv2.getOptimalNewCameraMatrix(camMat, dist, shape, 1.0, (1280, 720))
	x, y, w, h = roi
	map1, map2 = cv2.initUndistortRectifyMap(camMat, dist, None, newcammtx, (1280, 720), cv2.CV_32FC1)
	newImg = cv2.remap( img, map1, map2, interpolation=cv2.INTER_LINEAR)
	newImg = newImg[y:y+h, x:x+w]
	cv2.imshow("new", newImg)
	cv2.imshow("old", img)
	key = cv2.waitKey(10)
	if key == 27 or key & 0xFF == ord('q'):
		break
	elif key == 32:
		cv2.imwrite("camres.png", newImg)

cap.release()		
cv2.destroyAllWindows()
