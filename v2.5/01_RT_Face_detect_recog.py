
# ----- Importations ----- #
import time
import os
import numpy as np
import cv2
import tflite_runtime.interpreter as tf

from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

from PIL import Image

# ----- Variables ----- #
embedder_model = 'Models/mobilefacenet.tflite'
classifyer_model = 'Models/classifyerV2_86.tflite'
facesFldr = 'Faces/'

# ----- Function to call in face detection loop ----- #
def recognition(face, regEmbeddings, names, embedder, inputE, outputE, classifyer, inputI, outputI):
    # Calculate embeddings
    embedder.set_tensor(inputE['index'], np.expand_dims(face, axis = 0))
    embedder.invoke()
    emb = embedder.get_tensor(outputE['index'])
    
    # Iterate comparison for all registered persons
    predictions = []
    for index in range(len(regEmbeddings)):
        classifyer.set_tensor(inputI[0]['index'], emb)
        classifyer.set_tensor(inputI[1]['index'], regEmbeddings[index])
        classifyer.invoke()
        pred = classifyer.get_tensor(outputI[0]['index'])[0]
        predictions.append(pred)
    
    # return prediction
    if max(predictions)>=0.5: # possibilite en augmenetant de limiter les erreurs
        return names[predictions.index(max(predictions))], max(predictions)
    else:
        return 'Inconnu', -1


def main():
	# ----- Load models ----- #
	# Load mobilefacenet
	embedder = tf.Interpreter(embedder_model)
	embedder.allocate_tensors()
	inputE = embedder.get_input_details()[0]
	outputE = embedder.get_output_details()[0]

	# Load classifyer
	classifyer = tf.Interpreter(classifyer_model)
	classifyer.allocate_tensors()
	inputI = classifyer.get_input_details()
	outputI = classifyer.get_output_details()

	# ----- Prepare registered faces ----- #
	regEmbeddings = []
	names = []
	for regFaces in os.listdir(facesFldr):
		face = cv2.imread(os.path.join(facesFldr, regFaces))
		face = np.float32(face/255.)
		embedder.set_tensor(inputE['index'], np.expand_dims(face, axis = 0))
		embedder.invoke()
		emb = embedder.get_tensor(outputE['index'])
		regEmbeddings.append(emb)
		names.append(regFaces)

	# Initialize parameters
	font = cv2.FONT_HERSHEY_SIMPLEX
	width = 480
	height = 360
	fontScale = width/960 # 1 pour du 480p (480x960) Change with screen size
	colorName = (255, 0, 0) 
	colorBox = (0, 0, 255) 	# red BGR
	colorScore = (0, 255, 0) # green
	thickness_box = 1
	thickness_txt = 1
	newFps = 20

	# Open camera
	cap = cv2.VideoCapture(2) # find video number with v4l2-ctl --list-device
	
	# Définir la nouvelle résolution
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	print("Résolution actuelle :", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# Change fps
	fps = cap.get(cv2.CAP_PROP_FPS) # 30
	print("Frame rate before = {0}".format(fps))
	cap.set(cv2.CAP_PROP_FPS, newFps)
	fps = cap.get(cv2.CAP_PROP_FPS)
	print("Frame rate after = {0}".format(fps))

	# Create a window
	name_window = "Facial recognition"
	cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)

	# Information
	print("Press q to quit")

	# Initialize engine for detection
	engine = DetectionEngine('Models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite')
	
	somme = 0 # temps total des frames
	nbFrames = 0 # nombre total des frames
	debAppli = time.time()


	while True :
		# Debut mesure
		start = time.time()

		# Capture image
		ret, frame = cap.read()
		nbFrames += 1

		# Run inference.
		objs = engine.detect_with_image(Image.fromarray(frame),
					  threshold=0.05,
					  keep_aspect_ratio=True,
					  relative_coord=False,
					  top_k=10)

		# Print and draw detected objects.
		if len(objs) > 0 :
			for obj in objs:
				box = obj.bounding_box.flatten().tolist()
				# from float to int
				x1 = int(box[0])
				y1 = int(box[1])
				x2 = int(box[2])
				y2 = int(box[3])

				# Draw the box
				distX = abs(x2-x1)
				distY = abs(y2-y1)

				# Redefinition fontScale in function of box size
				fontScale = round(width/100) * distX*distY/(width*height)

				# box with 2 lines in each corner
				frame = cv2.line(frame, (x1,y1), (int(x1+distX/4),y1), colorBox, thickness_box)
				frame = cv2.line(frame, (x1,y1), (x1,int(y1+distY/4)), colorBox, thickness_box)

				frame = cv2.line(frame, (x2,y2), (x2,int(y2-distY/4)), colorBox, thickness_box)
				frame = cv2.line(frame, (x2,y2), (int(x2-distX/4),y2), colorBox, thickness_box)

				frame = cv2.line(frame, (x1,y2), (int(x1+distX/4),y2), colorBox, thickness_box)
				frame = cv2.line(frame, (x1,y2), (x1,int(y2-distY/4)), colorBox, thickness_box)

				frame = cv2.line(frame, (x2,y1), (x2,int(y1+distY/4)), colorBox, thickness_box)
				frame = cv2.line(frame, (x2,y1), (int(x2-distX/4),y1), colorBox, thickness_box)
				#frame = cv2.rectangle(frame, (x1,y1), (x2,y2), colorBox, thickness_box) # traditional box

				face=frame[y1:y2,x1:x2,:] # extract the face
				face = np.float32(cv2.resize(face, (112,112))/255.) # resize the face as a recognition input

				# Recognition
				name,score =recognition(face, 
										regEmbeddings, 
										names,
										embedder, 
										inputE, 
										outputE, 
										classifyer, 
										inputI, 
										outputI)
				# Draw the text
				name = name.split(".",2)[0]
				frame = cv2.putText(frame, name, (x1,y1-thickness_box-5), font, fontScale, colorName, thickness_txt, cv2.LINE_AA)

				# Draw the score
				if(score!=-1):
					score[0]=round(score[0],2)
					texteSize = cv2.getTextSize(str(score[0]), font, fontScale, thickness_txt) # get the score size
					frame = cv2.putText(frame, str(score[0]), (x2-texteSize[0][0],y1-thickness_box-5), font, fontScale, colorScore , thickness_txt, cv2.LINE_AA) # put the end of the text at the top right corner
			
		# Show the video input	
		cv2.imshow(name_window, frame)

		end = time.time()
		somme +=  end-start

		# Wait a pressed key 	
		key = cv2.waitKey(1)

		if key == ord("q") :
			print("Average processing time for 1 frame = ", round(somme/nbFrames,3), "s (",nbFrames,"frames )")
			break

	finAppli = time.time()
	print("Capture time =",round(finAppli-debAppli,3),"s")

	# Free cam resources
	cap.release()

	# Close Windows
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
