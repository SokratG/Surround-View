import numpy as np
import cv2, PIL, os, sys
from cv2 import aruco

argCnt = len(sys.argv)

if argCnt != 3:
  print("Not enough arguments")
  print("Example: calibration.py /path/to/dir/with/photos devName")
  sys.exit()

workdir = sys.argv[1]
devName =  sys.argv[2]

if not os.path.isdir(workdir):
  print("Invalid directory:", workdir)
  sys.exit()

print("Work Dir:", workdir)
print("Device Name:", devName)

arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
board  = aruco.CharucoBoard_create(10, 7, 54, 35, arucoDict)
#board  = aruco.CharucoBoard_create(9, 6, 45, 26, arucoDict)

images = [os.path.join(workdir,f) for f in os.listdir(workdir) if f.endswith(".png")]

def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, arucoDict)
        
        if len(corners)>0:
            # SUB PIXEL DETECTION
            cv2.imwrite("hello.png", gray)
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (5,5),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)

            retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            print(retval)
            if retval != 0:
                allCorners.append(charucoCorners)
                allIds.append(charucoIds)

    imsize = gray.shape
    return allCorners, allIds, imsize


allCorners, allIds, imsize = read_chessboards(images)

def calibrate_camera(allCorners, allIds, imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[ 2000.,    0., imsize[0] / 2.],
                                 [    0., 2000., imsize[1] / 2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners,allIds,imsize)

print("ERROR", ret)
print("K", mtx)
print("dist", dist)

np.savetxt(devName + ".K", mtx, fmt="%0.8f")
np.savetxt(devName + ".dist", dist, fmt="%0.8f")
