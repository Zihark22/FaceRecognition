# Take a screenshot with USB camera and OpenCV

import cv2

# Open camera
cap = cv2.VideoCapture(2) # find video number with v4l2-ctl --list-device

# Create a window
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)


while True :
	# Capture image
	ret, frame = cap.read()
	
	# Show the video input	
	cv2.imshow("Video", frame)
	
	# Wait a pressed key 	
	key = cv2.waitKey(1)

	if key == ord(" "):
		# Save image
		cv2.imwrite("image.jpg", frame)
		print("Image captured")
		break
	elif key == ord("q") :
		break
			
# Free cam resources
cap.release()

# Close Windows
cv2.destroyAllWindows()

