import argparse
import cv2
import time
import serial
import numpy as np


asd=10
bsd=10
csd=10

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation = inter)
	
def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def classifyBasic(img):
	(h,w) = img.shape[:2]
	right = img[0:h,w/2:w]
	left = img[0:h, 0:w/2]
	if cv2.countNonZero(left) > cv2.countNonZero(right):
		return 0
	return 1 


def houghLines(edges,img):
	minLineLength = 200
	maxLineGap = 20
	(h,w) = edges.shape[:2]
	edges_new = edges[6*h/8:7*h/8,0:w]
	new_img = img[6*h/8:7*h/8,0:w]
	(h,w) = edges_new.shape[:2]
 	lines = cv2.HoughLinesP(edges_new, asd ,np.pi/180,10,minLineLength,bsd)
	
	if isinstance(lines,np.ndarray):
		for x1,y1,x2,y2 in lines[0]:
		    cv2.line(new_img,(x1,y1),(x2,y2),(0,255,0),2)

	return new_img


def hough2(edges,img):
	(h,w) = edges.shape[:2]
	edges_new = edges[6*h/8:7*h/8,0:w]
	#new_img = img[6*h/8:7*h/8,0:w]
	#(h,w) = edges_new.shape[:2]
	leftBorder=0
	rightBorder=0
	lines = cv2.HoughLines(edges_new,2,np.pi/180,10)
	if isinstance(lines,np.ndarray):
		for rho,theta in lines[0]:
			if theta >= 0.85 and theta <= 1.0:
				leftBorder=1
			if theta >= 2.16 and theta <= 2.2:
				rightBorder=1

			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			#cv2.line(new_img,(x1,y1),(x2,y2),(0,0,255),2)
	return (leftBorder,rightBorder)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-c", "--camera", help="camera index", required=False, type=int, default=0)
ap.add_argument("-p", "--port", help="com port to communicate", default="/dev/rfcomm0")
args = vars(ap.parse_args())

ser = serial.Serial(
	port=args["port"],
	baudrate=9600
)

if not ser.isOpen():
	ser.open()

cIndex = args["camera"];

if not args.get("video", False):
	camera = cv2.VideoCapture(cIndex)
else:
	camera = cv2.VideoCapture(args["video"])


# Coefficients

out_index=0

bigSquareMin=220
bigSquareMax=300

mediumSquareMin=180
mediumSquareMax=219

smallSquareMin=30
smallSquareMax=80

blurKernel1=5
blurKernel2=3

edgeMin=180
edgeMax=280

blackBoxMin = 180
blackBoxMax = 220

#states
moving=0
rotating=0;
found=0

left=-1


while True:
	(grabbed, frame) = camera.read()
	
	if not grabbed:
		break

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (blurKernel1, blurKernel1), 0)
	_, thresholded = cv2.threshold(blurred,40,255,cv2.THRESH_BINARY)
	(h,w) = thresholded.shape[:2];
	thresholded = thresholded[3*h/5:5*h/6, 40:w-40]
	edged = cv2.Canny(blurred, edgeMin, edgeMax)
	edged_blur = edged.copy()
	edged_blur = cv2.GaussianBlur(edged_blur, (blurKernel2,blurKernel2), 0)
	kernel = np.ones((5,5), np.uint8)
	dilated = cv2.dilate(edged_blur, kernel, iterations=3)
	img_erosion = cv2.erode(dilated, kernel, iterations=3)

	thresholded_erosion = cv2.erode(thresholded,kernel,iterations=4)
	thresholded_erosion = cv2.dilate(thresholded_erosion,kernel, iterations=6)

	(_, cnts2, _) = cv2.findContours(cv2.bitwise_not(thresholded_erosion), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	out_frame = [frame,edged,thresholded, thresholded_erosion]

	found=0
	contoursCount=0
	currentPos=0

	for c in cnts2:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.01 * peri, True)
		(x,y,w,h) = cv2.boundingRect(approx);
		#cv2.drawContours(out_frame[out_index], [approx], -1, (0, 255, 255), 4)
		currentPos = (x+w)/2.0;
		print currentPos


	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.01 * peri, True)

		#		currentPos = (x+w)/2
		#		print currentPos

		
		if len(approx) >= 4 and len(approx) <= 6:
			(x, y, w, h) = cv2.boundingRect(approx)
			aspectRatio = w / float(h)
			area = cv2.contourArea(c)
			hullArea = cv2.contourArea(cv2.convexHull(c))
			solidity = area / float(hullArea)

			keepDimsMedium = w > mediumSquareMin and h > mediumSquareMin and w < mediumSquareMax and h < mediumSquareMax
			keepDimsBig = w > bigSquareMin and h > bigSquareMin and w < bigSquareMax and h < bigSquareMax
			keepDimsSmall = w > smallSquareMin and h > smallSquareMin and w < smallSquareMax and h < smallSquareMax

			keepSolidity = solidity > 0.9
			keepAspectRatio = aspectRatio > 0.88 and aspectRatio < 1.12

			
			if keepDimsMedium and keepSolidity and keepAspectRatio:
				cv2.drawContours(out_frame[out_index], [approx], -1, (0, 255, 0), 4)
				contoursCount+=1
				foundMedium=1
				out_frame[2] = four_point_transform(gray,approx.reshape(4,2))
				rett, cleanImg = cv2.threshold(out_frame[2],80,255,cv2.THRESH_BINARY)
				(h, w) = cleanImg.shape[:2]
				out_frame[3]=cleanImg[10:h-10, 10:w-10]
				left = classifyBasic(out_frame[3])
				#out_frame[3]=cleanImg[0:h, w/2:w]
				#out_frame[2]=cleanImg[0:h, 0:w/2]
			if keepDimsBig and keepSolidity and keepAspectRatio:
				cv2.drawContours(out_frame[out_index], [approx], -1, (0, 0, 255), 4)
				contoursCount+=1
				found=1

	# draw the status text on the frame
	if found and moving:
		ser.write("S")
		moving=0
		#left = classifyBasic(out_frame[3])
		if left == 0:
			ser.write("R")
			rotating=1
		if left == 1:
			ser.write("L")
			rotating=1

	if rotating:
		#l,r = hough2(img_erosion,frame.copy())
		
		#if l or r:
		#	ser.write("S")
		#	rotating=0
		if currentPos >blackBoxMin and currentPos <blackBoxMax:
			ser.write("S")
			rotating=0


	
	cv2.putText(out_frame[out_index], str(contoursCount)+" left: "+str(left) + " small square : "+ str(smallSquareMin) + ","  + str(smallSquareMax), 
		(20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		(0, 0, 255), 2)

	# show the frame and record if a key is pressed
	cv2.imshow("Frame", out_frame[out_index])
	key = cv2.waitKey(1) & 0xFFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

	if key == 83:
		ser.write("R")
		moving=1

	if key == 81:
		ser.write("L")
		moving=1

	if key == 82:
		ser.write("U")
		moving=1

	if key == 84:
		ser.write("D")
		moving=1


	if key == ord("s"):
		ser.write("S")
		moving=0
		rotating=0

	if key == ord("r"):
		ser.write("U")
		moving=1

	if key == ord ("b"):
		ser.write("D")
		moving=1


	if key == ord("f"):
		if left == 0:
			ser.write("R")
			rotating=1
		if left == 1:
			ser.write("L")
			rotating=1

	if key == ord("m"):
		out_index+=1
		if out_index>=4:
			out_index-=4

	if key == ord("["):
		smallSquareMin-=10
	if key == ord("]"):
		smallSquareMin+=10
	if key == ord("'"):
		smallSquareMax-=10
	if key == ord("\\"):
		smallSquareMax+=10

	#if key == ord(","):
	#	#asd-=1
	#if key == ord("."):
	#	#asd+=1

# cleanup the camera and close any open windows
#ser.close()
camera.release()
cv2.destroyAllWindows()