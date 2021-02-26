import numpy as np 
import cv2
import argparse
import imutils


def corner(points):
	rect = [0]*4
	pt_sum = np.sum(points, axis=1)

	rect[0] = points[np.argmin(pt_sum)]
	rect[2] = points[np.argmax(pt_sum)]
	
	pt_diff = np.diff(points, axis=1)

	rect[1] = points[np.argmin(pt_diff)]
	rect[3] = points[np.argmax(pt_diff)]
	
	return rect




ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required=True, help='path to the image')

args = vars(ap.parse_args())

image_path = args['image']


image = cv2.imread(image_path)

answers = {1:'B',2:'E',3:'A',4:'C',5:'C'}

label = {'A':1,'B':2,'C':3,'D':4,'E':5}

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray,(3,3),0)

can = cv2.Canny(blur,30,150)



cnt,_= cv2.findContours(can.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


c = max(cnt, key = cv2.contourArea)

peri = cv2.arcLength(c, True)
c = cv2.approxPolyDP(c, 0.02*peri, True)

points = [tuple(point[0]) for point in c]

points = corner(points)

points = np.array(points)

h1 = np.linalg.norm(points[0]-points[3])
h2 = np.linalg.norm(points[1]-points[2])

w1 = np.linalg.norm(points[0]-points[1])
w2 = np.linalg.norm(points[2]-points[3])

maxh = max(int(h1),int(h2))
maxw = max(int(w1),int(w2))

dest = [(0,0),(maxw-1,0),(maxw-1,maxh-1),(0,maxh-1)]

M = cv2.getPerspectiveTransform(np.array(points,dtype='float32'),np.array(dest,dtype='float32'))

warped = cv2.warpPerspective(image,M,(maxw,maxh))


g1 = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
b1 = cv2.GaussianBlur(g1, (3,3), 0)
ret, thresh = cv2.threshold(b1, 190, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3,3), dtype='int8')


#dilate = cv2.dilate(thresh, kernel, iterations=5)
erode = cv2.erode(thresh, kernel, iterations=5)
can1 = cv2.Canny(thresh,30,150)

cn,x = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = warped.copy()


c1 = {}
for c in cn:
	(x,y,w,h) = cv2.boundingRect(c)
	ar = w/ float(h)
	
	if w>= 20 and h>= 20 and ar >= 0.9 and ar<= 1.1:
		M = cv2.moments(c)
		(x,y) = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
		center = (x,y)
		c1[center] = c
		
		

			
centers = sorted(c1, key=lambda x: [x[1],x[0]])

count = 0
counter = 1
bubbled = None
bubbles = {}
for center in centers:

	mask = np.zeros(thresh.shape,dtype='uint8')
	cv2.drawContours(mask, [c1[center]], 0, 255, -1)
	mask = cv2.bitwise_and(thresh, thresh, mask=mask)
	count = cv2.countNonZero(mask)
	if bubbled is None or count > bubbled[0]:
		bubbled = (count,counter,center)
	
	if counter %5==0:
		count = 0
		counter = 0
		bubbles[bubbled[2]] = bubbled[1]
		bubbled = None
	counter += 1
	

counter = 1
count = 1
right = 0
wrong = 0
for center in centers:
	b = bubbles.get(center,None)
	if b is not None:
		print(label[answers[counter]])
		if label[answers[counter]] == b:
			right += 1
			cv2.drawContours(output, [c1[center]], 0, (0,255,0), 2)
		else:
			wrong += 1
			cv2.drawContours(output, [c1[center]], 0, (0,0,255), 2)
	count += 1
	if count %5 == 1:
		counter += 1	

score = right*100/(right+wrong)


cv2.putText(output, '{}%'.format(score), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2)
	
cv2.imshow('im',output)
cv2.waitKey(0)		
		
