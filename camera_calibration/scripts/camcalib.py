import numpy as np
import cv2 as cv
import sys
import os



argCnt = len(sys.argv)

if argCnt != 2:
	print("Not enough arguments")
	print("Example: capture.py \"camera number\"")
	sys.exit()
chh, chw = 9, 6

# termination criteria
#criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria= (cv.TERM_CRITERIA_EPS & cv.TERM_CRITERIA_COUNT, 10000, 1e-9)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chh*chw,3), np.float32)
objp[:,:2] = np.mgrid[0:chh,0:chw].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

resolutionX = 1920
resolutionY = 1080
max_Calib_samples = 12
min_Calib_samples = 10
chbSize = (chh, chw)

numdevName = int(sys.argv[1])

video_device = [0, 1, 2, 3]
devName = "video" + str(video_device[numdevName])
gSource = "v4l2src"
gDevice = " device=/dev/" + devName
gPipline = " ! video/x-raw,width={:d},height={:d},format=(string)UYVY ! nvvidconv ! video/x-raw(memory:NVMM),width={:d},height={:d},format=(string)I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink".format(resolutionX, resolutionY, resolutionX, resolutionY)

gCapture = gSource + gDevice + gPipline

cap = cv.VideoCapture(gCapture)

count_calib = 0
ret, img = cap.read()
h, w, c = img.shape
shape = (h, w)

while(cap.isOpened()):
    ret, img = cap.read() #cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if ret == True:
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chbSize, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria) #
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(gray, chbSize, corners2, ret)
            count_calib += 1
    if count_calib >= max_Calib_samples:
        break	
    cv.imshow('img', gray)
    key = cv.waitKey(5)
    if key == 27 or key & 0xFF == ord('q'):
        break
cap.release()		
cv.destroyAllWindows()

if count_calib >= min_Calib_samples:
	flags = (cv.CALIB_USE_INTRINSIC_GUESS + cv.CALIB_RATIONAL_MODEL)
	distCoeffsInit = np.zeros((5,1))
	cameraMatrixInit = np.array([[1000.0, .0, shape[0]/2], [.0, 1000.0, shape[1]/2],[0.0, 0.0, 1.]])
	ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints=objpoints, imagePoints=imgpoints, imageSize=shape, cameraMatrix=cameraMatrixInit, distCoeffs=distCoeffsInit, flags=flags, criteria=criteria)

	np.savetxt(devName + ".K", mtx, fmt="%0.8f")
	np.savetxt(devName + ".dist", dist, fmt="%0.8f")
	"""
	newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
	dst = cv.undistort(gray, mtx, dist, None, newcameramtx)
	x, y, w, h = roi
	dst = dst[y:y+h, x:x+w]
	cv.imwrite("result_unidst.png", dst)

	mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
	dst = cv.remap(gray, mapx, mapy, cv.INTER_LINEAR)
	x, y, w, h = roi
	dst = dst[y:y+h, x:x+w]
	cv.imwrite("result_remap.png", dst)
	"""

