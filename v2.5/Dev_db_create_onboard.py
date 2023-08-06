# Importations
import os
import cv2
import numpy as np

from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

from PIL import Image

# Initialize engine for detection
engine = DetectionEngine('ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite')

# Directory where the images are stored
image_dir = 'FaceDataset/'

# create a list to store the data
faces = []
names = []

# Initialise the max number of pictures and the counter
max_im = 2000
count = 0

# Initialise size (input size for our embedder)
size = (112,112)

# Iterate through each person's folder
for person in os.listdir(image_dir):
	person_dir = os.path.join(image_dir, person)
	if os.path.isdir(person_dir):
		# Iterate through each image in the folder
		for image_path in os.listdir(person_dir):
			if count<max_im:
				# Load the image
				image = cv2.imread(os.path.join(person_dir, image_path))
				# Run inference.
				objs = engine.detect_with_image(Image.fromarray(image),
							  threshold=0.05,
							  keep_aspect_ratio=True,
							  relative_coord=False,
							  top_k=10)
				maxScore = 0
				objPrincipal = objs[0]
				# find the principal face detected
				for obj in objs:
					score = obj.score
					if score > maxScore :
						maxScore = score
						objPrincipal = obj
				box = objPrincipal.bounding_box.flatten().tolist() # coordinates of the face
				# foat to int
				x1 = int(box[0])
				y1 = int(box[1])
				x2 = int(box[2])
				y2 = int(box[3])
				face=image[y1:y2,x1:x2,:]  # extract the face
				face_image = cv2.resize(face,size) #resize the image
				faces.append(face_image) # add face
				names.append(person) # add name
				count += 1
				print(count)

# Save the data to .npy format
np.save('DevDatasets/faces.npy', np.array(faces))
np.save('DevDatasets/names.npy', np.array(names))


