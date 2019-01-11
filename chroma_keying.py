# import cv2

# img = cv2.imread("1.jpeg",1)
# bg = cv2.imread("2.jpeg",1)

# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# m = len(img)
# n = len(img[0])

# bg = cv2.resize(bg,(n, m),interpolation = cv2.INTER_AREA) 

# # for i in range(m):
# # 	for j in range(n):
# # 		print(str(hsv_img[i][j][0]) + " ")
# # 	print("\n")
# # cv2.imshow("image",hsv_img)
# # cv2.waitKey(0)

# for i in range(m):
# 	for j in range(n):
# 		if(img[i][j][0]>=0 and img[i][j][0]<=100 and img[i][j][1]>=125 and img[i][j][1]<=280 and img[i][j][2]>=15 and img[i][j][2]<=100):
# 			img[i][j][0] = bg[i][j][0]
# 			img[i][j][1] = bg[i][j][1]
# 			img[i][j][2] = bg[i][j][2]
# 		# if(hsv_img[i][j][0]>=30 and hsv_img[i][j][0]<=60):
# 		# 	img[i][j][0] = bg[i][j][0]
# 		# 	img[i][j][1] = bg[i][j][1]
# 		# 	img[i][j][2] = bg[i][j][2]

# cv2.imshow("image",img)
# cv2.waitKey(0)

# lower_green = np.array([0,125,15])
# upper_green = np.array([100,280,100])

import numpy as np
import cv2
import os

cap = cv2.VideoCapture("f.mp4") # creating VideCapture object for the foreground video
back = cv2.VideoCapture("bg2.mov") # creating VideCapture object for the background video

fps = cap.get(cv2.CAP_PROP_FPS) # calculating frame rate of the foreground video

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, fps, (640,480)) # creating VideoWriter object to save the output video

while(cap.isOpened() and back.isOpened()):
    ret, img = cap.read() # reading frame of the foreground video
    re, bg = back.read() # reading frame of the background video
    if ret==True:
    	bg = cv2.resize(bg,(640, 480),interpolation = cv2.INTER_AREA)
    	img = cv2.resize(img,(640, 480),interpolation = cv2.INTER_AREA)
    	lower_green = np.array([0,110,0]) # setting lower range of [B,G,R] values
    	upper_green = np.array([120,280,120]) # setting upper range of [B,G,R] values
    	mask = cv2.inRange(img, lower_green, upper_green) # generating mask of the foreground frame
    	img[mask != 0] = [0, 0, 0] # applying mask to foreground frame
    	bg[mask == 0] = [0,0,0] # applying mask to background frame but in opposite way
    	final = bg + img # creating final image
    	cv2.imshow('frame',final)
    	out.write(final)
    	if cv2.waitKey(1) & 0xFF == ord('q'):
    		break
    else:
    	break

cap.release()
out.release()
cv2.destroyAllWindows()
