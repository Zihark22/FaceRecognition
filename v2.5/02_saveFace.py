# ----- Importations ----- #
import numpy as np
import cv2

from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

from PIL import Image

dossier_faces = 'Faces/'

def main():
	# Open camera
	cap = cv2.VideoCapture(2) # find video number with v4l2-ctl --list-device

	# Create a window
	name_window = "Save face"
	cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)

	# Information
	print("Press q to quit and space to capture a picture")

	# Initialize parameters
	colorBox = (0, 0, 0) # black
	thickness_box = 1

	# Initialize engine for detection
	engine = DetectionEngine('Models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite')

	while True :

		# Capture image
		ret, frame = cap.read()
		frameRect = frame

		# Run inference to detect faces
		objs = engine.detect_with_image(Image.fromarray(frame),
					  threshold=0.05,
					  keep_aspect_ratio=True,
					  relative_coord=False,
					  top_k=10)

		# Print and draw detected objects.
		if len(objs) == 1 : # check if only one face was detected
			obj = objs[0] # take the first face
			box = obj.bounding_box.flatten().tolist() # take the box coordinates
			# float to int
			x1 = int(box[0])
			y1 = int(box[1])
			x2 = int(box[2])
			y2 = int(box[3])
			
			# Draw the box
			face=frame[y1:y2,x1:x2,:] # take the face inside the picture
			face = cv2.resize(face, (112,112)) # resize it in 112*112 for the embedder
			frameRect = cv2.rectangle(frameRect, (x1,y1), (x2,y2), colorBox, thickness_box) # draw the box

		# Show the video input	
		cv2.imshow(name_window, frameRect)

		# Wait a pressed key 	
		key = cv2.waitKey(1)

		if key == ord("q") : # if q pressed : leave the loop and end
			break
		elif key == ord(" ") :
			if len(objs) == 1 :
				print("Enter your name : ")
				name = input()   # wait for input (name)
				cv2.imwrite(dossier_faces+name+".jpg", face) # save the picture in database
				print("Image captured")
		
	# Free cam resources
	cap.release()

	# Close Windows
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()