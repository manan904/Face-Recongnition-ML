import numpy as np
import cv2

#Camera Support
cam=cv2.VideoCapture(0)

#Classifier for face detection(harr-cascade classifier decides what all features to extract)
face_cas = cv2.CascadeClassifier('/Users/mananmanwani/Downloads/ml-webinar-face-recoginition-knn-master 2/haarcascade_frontalface_alt.xml')

#Stores feature data
data =[]	
ix=0 #Total number of frames  captured

while True:
	#Ret=Return Value(bool) to check whether the camera is working or not!
	ret,frame = cam.read()
	if ret==True:
		#Converting the image into grayscale!
		gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		#Applying the classifier to detect faces in the current frame!
		faces=face_cas.detectMultiScale(gray,1.3,5) 

		#We get 4 components from each face
		# x,y are the corner coordinates and h,w are height and width of the face frame respectively
		for(x,y,w,h) in faces:

			#Get the face component
			face_component =frame[y:y+h, x:x+w, :]
			#Resize the face component
			fc= cv2.resize(face_component,(50,50))

			#Store face data after every 10 frames and do this till we have 20 such face entries
			if ix%10==0 and len(data)<20:
				data.append(fc)

			#For Visualisation purposes,we add a rectangular boundary	
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		ix+=1
		#Show the frame
		cv2.imshow('frame',frame)
		print len(data)
		#Break loop when Esc key is pressed or 20 entries have been taken
		if cv2.waitKey(1)== 27 or len(data) >= 20:
				break

	else:
		print 'error'

#Destory the windows opened for face recognition
cv2.destroyAllWindows()

#Convert Data into an ns array
data=np.asarray(data)

print data.shape
#Save face data as a numpy matrix
np.save('face_01',data)