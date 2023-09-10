import numpy as np
import cv2 as cv
import os
def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())
def knn (train,test,k=5):
    dist=[]
    for i in range(train.shape[0]):
        ix=train[i,:-1]
        iy=train[i,-1]
        d=distance(test,ix)
        dist.append([d,iy])
    dk=sorted(dist,key=lambda x: x[0])[:k]
    labels=np.array(dk)[:,-1]
    output=np.unique(labels,return_counts=True)
    index=np.argmax(output[1])
    return output[0][index]
cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")
dataset_path="./face_dataset/"
face_data=[]
labels=[]
class_id=0
names= {}
#data preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id]=fx[:-4]
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)
        target=class_id*np.ones((data_item.shape[0],))
        class_id=+1
        labels.append(target)
face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))
print(face_labels.shape)
print(face_dataset.shape)

trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)
font=cv.FONT_HERSHEY_SIMPLEX


while True:
    ret, frame = cap.read()
    if ret==False:
        continue
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for face in faces:
        x,y,w,h=face
        offset=5
        face_section=frame[y-offset: y+h+offset,x-offset:x+w+offset]
        face_section=cv.resize(face_section,(100,100))
        out=knn(trainset,face_section.flatten())
        cv.putText(frame,names[int(out)],(x,y-10),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv.LINE_AA)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv.imshow("Faces",frame)
    if cv.waitKey(1)& 0xff == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


