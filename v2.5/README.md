# Ptech

## Face detection and recognition on a google coral card

Julien Badajoz & Alan Gerard | ETN5 | SMTR

## Version 2.5
Adapt text size to screen

## Version 2.4
Change resolution to reduce latency

## Version 2.3
Add some comments and change langage to english

## Version 2.2
Path access modified for 02_saveFace.py with new file system tree

## Version 2.1
Box's design modified and Paul added to Faces/

### Version: 2.0
New classifyer trained with faces cropped by the same model running on the board 86% accuracy.

---
## Goal
Create a real time face detection and recognition system to run on a google coral card. Adding a person to the database should not require retraining the model.

## Roadmap
- [x] Face detection using MTCNN, handmade NN or pretrained model  
- [x] Embeddings extraction from faces using FaceNet
- [x] Embeddings comparison using siamese Networks
- [x] Assemble the hole system
- [x] Use it with the google coral card
- [x] Create the real time application
- [ ] Test and optimisation of the application

## Architecture of the final system

1. Real_time_image -> Face detection -> Faces
2. Faces -> Embeddings extraction
3. Registered_faces -> Embeddings extraction
4. Embeddings -> Embeddings comparison -> recognition
5. Display of boxes of detected faces and the result of their recognition

---

## Libraries

### For developpement:

- Python 3.10
- Tensorflow 2.10
- Keras
- Numpy
- os
- cv2
- matplotlib
- sklearn
- mtcnn
- random

### For usage on google coral:
- Python 3.7
- time
- os
- numpy
- cv2
- tflite_runtime
- edgetpu
- PIL