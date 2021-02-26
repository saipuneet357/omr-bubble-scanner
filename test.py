import cv2
import imutils
import numpy as np
import argparse




ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input dataset of images")
args = vars(ap.parse_args())


img = cv2.imread(args['image'])

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


blur = cv2.GaussianBlur(gray,(5,5),0)


ret,thresh = cv2.threshold(blur,146,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

print(ret)	
#can = cv2.Canny(blur,0,100)

kernel = np.ones((3,3),dtype='int8')


dilate = cv2.dilate(thresh,kernel,iterations=5) 

#img = cv2.bitwise_and(img,img,mask=dilate)
erode = cv2.erode(dilate,kernel,iterations=5) 

can = cv2.Canny(erode,30,150)

cnts = cv2.findContours(can.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) 
cnts = imutils.grab_contours(cnts)

output = img.copy()

centers = {}

for c in cnts:
	epsilon = 0.03*cv2.arcLength(c,True)
	approx = cv2.approxPolyDP(c,epsilon,True)
	(x,y),radius = cv2.minEnclosingCircle(approx)
	center = (int(x),int(y))
	radius = int(radius)
	if centers.get(center,None) == None:
		centers[center] = radius
	

	
print(centers)

for center in  list(centers.keys())[:-1]:
	cv2.circle(output,center,centers[center],(0,255,0),2)
	


while True:
	cv2.imshow('image',output)
	if cv2.waitKey(0) & 0b11111111 == ord('q'):
		break

for c in cn:
	epsilon = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.03*epsilon, True)
	(x,y),radius = cv2.minEnclosingCircle(approx)
	cv2.circle(output, (int(x),int(y)), int(radius), (0,255,0), 2)
	#cv2.drawContours(output, [c], 0, (0,255,0), 2)
	cv2.imshow('out',output)
	if cv2.waitKey(0) & 0b11111111 == ord('q'):
		break



