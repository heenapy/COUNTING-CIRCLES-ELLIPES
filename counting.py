# ___________-COUNTING CIRCLES AND ELLIPES____________________
import cv2
import numpy as np
image = cv2.imread('sh2.png')
cv2.imshow('original',image)
cv2.waitKey()

detector = cv2.SimpleBlobDetector_create()

keypoints = detector.detect(image)

blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image,keypoints,blank,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)

text = "total number of blobs:" + str(len(keypoints))

cv2.putText(blobs, text, (20,550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)
cv2.imshow('blobs using default parameters',blobs)
cv2.waitKey()

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea =100

params.filterByCircularity = True
params.minConvexity = 0.2

params.filterByInertia = True
params.minInertiaRatio = 0.1

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(image)
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image,keypoints,blank,(0,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)

text = "number of circuler blobs:" + str(len(keypoints))

cv2.putText(blobs, text, (20,550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)
cv2.imshow('filtering circular blobs only',blobs)
cv2.waitKey()

cv2.destroyAllWindows()