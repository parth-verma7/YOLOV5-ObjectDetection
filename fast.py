import numpy as np
import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n6.pt')
# print(model.names)
# model.names[0]
size=416
color=(0,0,255)

cap=cv2.VideoCapture(0)
while True:
    ret,img=cap.read()
    results=model(img,size)
    # print(results)
    for index,row in results.pandas().xyxy[0].iterrows():
        x1=int(row['xmin'])
        x2=int(row['xmax'])
        y1=int(row['ymin'])
        y2=int(row['ymax'])
        d=row['class']
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
        rectx1,recty1=(x1+x2)/2,(y1+y2)/2
        rectcenter=int(rectx1),int(recty1)
        cx=rectcenter[0]
        cy=rectcenter[1]
        cv2.putText(img,str(model.names[d]),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
        print(model.names[d])
    cv2.imshow("IMG",img)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()


