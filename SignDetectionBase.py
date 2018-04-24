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

out_index=0
blurx=220
blury=290

blur1=5
moving=0
rotating=0;
found=0

edge1=180
edge2=280

tri1 = 50
tri2 = 70

left=-1


while True:
	(grabbed, frame) = camera.read()
	status = "No Targets"
	
	if not grabbed:
		break

	# convert the frame to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (blur1, blur1), 0)
	#_, thresholded = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
	edged = cv2.Canny(blurred, edge1, edge2)
	edged_blur = edged.copy()
	edged_blur = cv2.GaussianBlur(edged_blur, (3,3), 0)
	kernel = np.ones((5,5), np.uint8)
	dilated = cv2.dilate(edged_blur, kernel, iterations=3)
	img_erosion = cv2.erode(dilated, kernel, iterations=3)

	#cnts = cv2.findContours(edged_blur.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

	# find contours in the edge map
	(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	out_frame = [frame,edged,dilated, blurred]

	found=0
	contoursCount=0
	currentPos=0
	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.01 * peri, True)

		if len(approx) == 4:
			(x,y,w,h) = cv2.boundingRect(approx)
			keepDims = w >= tri1 and w<= tri2 and h < tri2
			if keepDims:
				cv2.drawContours(out_frame[out_index], [approx], -1, (0, 0, 255), 4)
				currentPos = (x+w)/2
				print currentPos

		# ensure that the approximated contour is "roughly" rectangular
		if len(approx) >= 4 and len(approx) <= 6:
			# compute the bounding box of the approximated contour and
			# use the bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			aspectRatio = w / float(h)

			# compute the solidity of the original contour
			area = cv2.contourArea(c)
			hullArea = cv2.contourArea(cv2.convexHull(c))
			solidity = area / float(hullArea)

			# compute whether or not the width and height, solidity, and
			# aspect ratio of the contour falls within appropriate bounds
			keepDims = w > blurx and h > blurx and w < blury and h < blury
			keepSolidity = solidity > 0.9
			keepAspectRatio = aspectRatio >= 0.85 and aspectRatio <= 1.15

			# ensure that the contour passes all our tests
			if keepDims and keepSolidity and keepAspectRatio:
				# draw an outline around the target and update the status
				# text
				cv2.drawContours(out_frame[out_index], [approx], -1, (0, 0, 255), 4)
				status = "Target(s) Acquired"
				contoursCount+=1
				found=1
				out_frame[2] = four_point_transform(gray,approx.reshape(4,2))
				rett, cleanImg = cv2.threshold(out_frame[2],80,255,cv2.THRESH_BINARY)
				(h, w) = cleanImg.shape[:2]
				out_frame[3]=cleanImg[15:h-15, 15:w-15]
				#out_frame[3]=cleanImg[0:h, w/2:w]
				#out_frame[2]=cleanImg[0:h, 0:w/2]
				# compute the center of the contour region and draw the
				# crosshairs
				M = cv2.moments(approx)
				(cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				(startX, endX) = (int(cX - (w * 0.15)), int(cX + (w * 0.15)))
				(startY, endY) = (int(cY - (h * 0.15)), int(cY + (h * 0.15)))
				cv2.line(out_frame[out_index], (startX, cY), (endX, cY), (0, 0, 255), 3)
				cv2.line(out_frame[out_index], (cX, startY), (cX, endY), (0, 0, 255), 3)

	# draw the status text on the frame
	if found and moving:
		ser.write("S")
		moving=0
		left = classifyBasic(out_frame[3])
	if rotating:
		#l,r = hough2(img_erosion,frame.copy())
		
		#if l or r:
		#	ser.write("S")
		#	rotating=0
		if currentPos >195 and currentPos <208:
			ser.write("S")
			rotating=0


	
	cv2.putText(out_frame[out_index], str(contoursCount)+" left: "+str(left) + " edge Coeff "+str(tri1)+" "+str(tri2)+" "+str(csd), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
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
		tri1-=10
	if key == ord("]"):
		tri1+=10
	if key == ord("'"):
		tri2-=10
	if key == ord("\\"):
		tri2+=10

	if key == ord(","):
		asd-=1
	if key == ord("."):
		asd+=1

# cleanup the camera and close any open windows
#ser.close()
camera.release()
cv2.destroyAllWindows()