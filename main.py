import cv2 as cv
import numpy as np
import pandas as pd
import PIL.Image as image
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os
def convert(path,class_name):
    images= [os.path.join(path, f) for f in os.listdir(path)]
    for img in images:
        img_read=image.open(img).convert('L')
        img_resize=img_read.resize((1024,1024))
        img_convert=np.array(img_resize).flatten()
        if class_name=='rock':
            data=np.append(img_convert,[0])
        elif class_name=='paper':
            data=np.append(img_convert,[1])
        elif class_name=='scissors':
            data=np.append(img_convert,[2])
        else:
            continue
        return [data]
rock=convert('./images/rock','rock')
paper=convert('./images/paper','paper')
scissors=convert('./images/scissors','scissors')
rock_df=pd.DataFrame(rock)
paper_df=pd.DataFrame(paper)
scissors_df=pd.DataFrame(scissors)
all_data=pd.concat([rock_df,paper_df,scissors_df])
inputVal=np.array(all_data)[:,:1048576]
outputVal=np.array(all_data)[:,1048576]
model=DecisionTreeClassifier()
clf=model.fit(inputVal,outputVal)
cap=cv.VideoCapture(0)
while True:
    _,frame=cap.read()
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (1024, 1024))
    img = np.array(img).flatten()
    pred=clf.predict([img])
    if pred==0:
        pred='rock'
    elif pred==1:
        pred='paper'
    elif pred==2:
        pred='scissors'
    frame=cv.putText(frame,pred,(0,185),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv.imshow('frame',frame)
    if cv.waitKey(1)==ord('q'):
        break
cap.release()
cv.destroyAllWindows()

