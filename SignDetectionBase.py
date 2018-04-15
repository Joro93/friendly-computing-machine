import argparse
import cv2
import time
import serial

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
blurx=180
blury=250

blur1=5
moving=0
rotating=0;
found=0


# keep looping
while True:
	(grabbed, frame) = camera.read()
	status = "No Targets"
	
	if not grabbed:
		break

	# convert the frame to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (blur1, blur1), 0)
	edged = cv2.Canny(blurred, 35, 150)
	edged_blur = edged.copy()
	edged_blur = cv2.GaussianBlur(edged_blur, (3,3), 0)

	#cnts = cv2.findContours(edged_blur.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

	# find contours in the edge map
	(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	out_frame = [frame,blurred,edged,edged_blur]

	found=0
	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.01 * peri, True)

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
			keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2

			# ensure that the contour passes all our tests
			if keepDims and keepSolidity and keepAspectRatio:
				# draw an outline around the target and update the status
				# text
				cv2.drawContours(out_frame[out_index], [approx], -1, (0, 0, 255), 4)
				status = "Target(s) Acquired"
				found=1

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
	if not found and rotating:
		ser.write("S")
		rotaing=0


	
	cv2.putText(out_frame[out_index], str(blurx)+" "+str(blury), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		(0, 0, 255), 2)

	# show the frame and record if a key is pressed
	cv2.imshow("Frame", out_frame[out_index])
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

	if key == ord("s"):
		ser.write("S")
		moving=0
		rotating=0

	if key == ord("r"):
		ser.write("U")
		moving=1;

	if key == ord("f"):
		ser.write("R")
		rotating=1	

	if key == ord("m"):
		out_index+=1
		if out_index>=4:
			out_index-=4
	if key == ord("["):
		blurx-=10
	if key == ord("]"):
		blurx+=10
	if key == ord("'"):
		blury-=10
	if key == ord("\\"):
		blury+=10

	if key == ord(","):
		blur1-=1
	if key == ord("."):
		blur1+=1

# cleanup the camera and close any open windows
ser.close()
camera.release()
cv2.destroyAllWindows()