import numpy as np
import cv2 as cv
import sys
import os
import glob

argCnt = len(sys.argv)

if argCnt != 2:
	print("Not enough arguments")
	print("Example: capture.py \"camera number\"")
	sys.exit()
chh, chw = 9, 6


# termination criteria
#criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
criteria= (cv.TERM_CRITERIA_EPS & cv.TERM_CRITERIA_COUNT, 1000, 1e-5)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chh*chw,3), np.float32)
objp[:,:2] = np.mgrid[0:chh,0:chw].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

resolutionX = 1920
resolutionY = 1080
max_Calib_samples = 14
min_Calib_samples = 10
chbSize = (chh, chw)

numdevName = int(sys.argv[1])
video_device = [0, 1, 2, 3]
devName = "video" + str(video_device[numdevName])


max_Calib_samples = 14
min_Calib_samples = 10


images = glob.glob('data/*.png')
count_calib = 0


for fname in images:
	img = cv.imread(fname, cv.IMREAD_GRAYSCALE) 
	shape = img.shape
	ret, corners = cv.findChessboardCorners(img, (chh, chw), None)

	if ret == True:
		objpoints.append(objp)
		corners2 = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
		imgpoints.append(corners2)
		count_calib += 1
		print('corners complete add')
	if count_calib >= max_Calib_samples:
        	break

if count_calib >= min_Calib_samples:
	flags = (cv.CALIB_USE_INTRINSIC_GUESS + cv.CALIB_RATIONAL_MODEL)
	distCoeffsInit = np.zeros((5,1))
	cameraMatrixInit = np.array([[1000.0, .0, shape[0]/2], [.0, 1000.0, shape[1]/2],[0.0, 0.0, 1.]])
else:
	print("Error calibrate camera - #" + str(numdevName))
	exit()

#ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints=objpoints, imagePoints=imgpoints, imageSize=shape, cameraMatrix=cameraMatrixInit, distCoeffs=distCoeffsInit, flags=flags, criteria=criteria)


img = cv.imread('data/video2_000.png') #add name image!
h, w, c = img.shape
shape = (w, h)

newcammtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, shape, 1, shape)
cv.imwrite("original.png", img)
x, y, w, h = roi

# undist
dst = cv.undistort(img, mtx, dist, None, newcammtx)
dst = dst[y:y+h, x:x+w]
cv.imwrite("undist.png", dst)

# remap
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcammtx, shape, 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
dst = dst[y:y+h, x:x+w]
cv.imwrite("remap.png", dst)



np.savetxt(devName + ".K", mtx, fmt="%0.8f")
np.savetxt(devName + ".dist", dist, fmt="%0.8f")















