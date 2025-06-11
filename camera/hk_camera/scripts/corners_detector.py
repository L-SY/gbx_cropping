import cv2

img = cv2.imread("/home/lsy/gbx_cropping_ws/src/camera/hk_camera/scripts/calibration_images/image.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
ret, corners = cv2.findChessboardCorners(gray, (5,5), flags)

if ret:
    cv2.drawChessboardCorners(img, (5,5), corners, ret)
    cv2.imshow("Corners", img)
    cv2.waitKey(0)
else:
    print("角点检测失败")
