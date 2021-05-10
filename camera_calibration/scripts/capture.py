import numpy as np
import cv2

import sys
import os

def exists(path):
    try:
        os.stat(path)
    except OSError:
        return False
    return True

argCnt = len(sys.argv)

if argCnt != 3 and argCnt != 5:
	print("Not enough arguments")
	print("Example: capture.py video0 /home/nvidia/Documents")
	print("Example: capture.py video0 /home/nvidia/Documents 2304 1536")
	sys.exit()

devName = sys.argv[1]
devPath = "/dev/" + devName
saveFolder = sys.argv[2]

resolutionX = 2304
resolutionY = 1536

if argCnt == 5:
    resolutionX = int(sys.argv[3])
    resolutionY = int(sys.argv[4])

if not exists(devPath):
	print("Invalid device:", devPath)
	sys.exit()

if not os.path.isdir(saveFolder):
	print("Invalid directory:", saveFolder)
	sys.exit()

print("Device path:", devPath)
print("Save folder:", saveFolder)
print('Resolution: {:d}x{:d}'.format(resolutionX, resolutionY))

gSource = "v4l2src"
gDevice = " device=" + devPath
gPipline = " ! video/x-raw,width={:d},height={:d},format=(string)UYVY ! nvvidconv ! video/x-raw(memory:NVMM),width={:d},height={:d},format=(string)I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink".format(resolutionX, resolutionY, resolutionX, resolutionY)

gCapture = gSource + gDevice + gPipline

cap = cv2.VideoCapture(gCapture)

imageCnt = 0

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        break
    elif key == 32:
        imgPath = os.path.join(saveFolder, devName + "_" + '{:03d}'.format(imageCnt) + "." + "png")
        cv2.imwrite(imgPath, frame)
        imageCnt+=1
        print("Save image:", imgPath)
	
cap.release()
cv2.destroyAllWindows()

