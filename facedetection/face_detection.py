import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
	ret,frame = cap.read()

	gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
	if len(faces) == 0:
		continue

	for face in faces[:1]:
		x,y,w,h = face

		offset = 10
		face_offset = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_selection = cv.resize(face_offset,(100,100))

		cv.imshow("Face", face_selection)
		cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

	cv.imshow("faces",frame)

	key_pressed = cv.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cap.release()
cv.destroyAllWindows()