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


resolutionX = 1920
resolutionY = 1080

numdevName = int(sys.argv[1])


video_device = [0, 1, 2, 3]
devName = "video" + str(video_device[numdevName])
gSource = "v4l2src"
gDevice = " device=/dev/" + devName
gPipline = " ! video/x-raw,width={:d},height={:d},format=(string)UYVY ! nvvidconv ! video/x-raw(memory:NVMM),width={:d},height={:d},format=(string)I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink".format(resolutionX, resolutionY, resolutionX, resolutionY)

gCapture = gSource + gDevice + gPipline
cap = cv.VideoCapture(gCapture)

imageCnt = 0

while(True):
    ret, frame = cap.read()
    cv.imshow('frame', frame)
    key = cv.waitKey(10)
    if key & 0xFF == ord('q') or key == 27:
        break
    elif key == 32:
        imgPath = os.path.join('data/', devName + "_" + '{:03d}'.format(imageCnt) + "." + "png")
        cv.imwrite(imgPath, frame)
        imageCnt+=1
        print("Save image:", imgPath)
	#print("image num - ", str(imageCnt) )

cap.release()
cv.destroyAllWindows()
