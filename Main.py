"""

1)Create folder and upload famous faces
2)Write folder's path in config file
3)Run!

"""
#Import some libraries
import cv2
import face_recognition
import os
#Open and read config
ConfFile = open("recognition.cfg", 'r')
PATH = ConfFile.read()
#Ad some variables 
knowFace = []
EncodingsFace = []
face_locations = []
rgb_frame = []
n = 0
unknown = 0
bs = ''
ukf = 0
listDir = os.listdir(PATH)											#Get list of famous faces
video_capture = cv2.VideoCapture(0)									#Capture webcam(zero in argument is any aviable camera)
#Download and encode famous faces
for i in range(len(listDir)):
	knowFace.append(face_recognition.load_image_file(PATH + listDir[i]))
	EncodingsFace.append(face_recognition.face_encodings(knowFace[i]))
#Main loop
while True:
	
	ret, frame = video_capture.read()								#Get frame
	rgb_frame = frame[:, :, ::-1]									#Invert color for recognition
	face_locations = face_recognition.face_locations(rgb_frame)		#Find faces in frame
	n = str(len(face_locations))									#Count faces
	unknown = face_recognition.face_encodings(rgb_frame)			#Encoding face from frame
	#Overlay frames on faces
	for top, right, bottom, left in face_locations:
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
	#Show count of faces on display
	cv2.putText(frame, n, (len(frame[0, ::, 0]) - 40,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	#Loop for compare face in frame and face in memory
	for j in range(len(unknown)):
		for i in range(len(EncodingsFace)-1):
			try:
				result = face_recognition.compare_faces(EncodingsFace[i], unknown[j])[0]	#Compare faces
			except IndexError:										#Litle kludge because I don’t know why this error appears
				continue
			print(result)
			if result == True:
				#Overlay people's name over his frame
				cv2.putText(frame, str(listDir[i])[:len(listDir[i]) - 5], (face_locations[j][3],face_locations[j][0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
				result = False
				break
	cv2.imshow('FaceRecognition', frame)							#Reload image
	face_locations = []												#Clear array of faces 
	#Exit for condition
	if cv2.waitKey(1) == ord('q'):
		break
video_capture.release()												#Disable Capture
cv2.destroyAllWindows()												#Delete all windows
