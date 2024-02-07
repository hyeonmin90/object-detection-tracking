# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:48:03 2023

@author: USER
"""

from datetime import datetime
from FunctionLibrary import *
import numpy as np
import cv2
import time

cam=cv2.VideoCapture("conveyor(3).mp4")


PTime=0


classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

CTime=time.time()
fps=1/(CTime-PTime)
PTime=CTime        

def detect(img):
    lower_range=np.array([0,0,0], dtype="uint8")
    upper_range=np.array([0,0,0], dtype="uint8")
    img=cv2.inRange(img,lower_range,upper_range)
    cv2.imshow("Range", img)
    m=cv2.moments(img)
    if(m["m00"] != 0) :
        x=int(m["m10"]/m["m00"])
        y=int(m["m01"]/m["m00"])
    else:
        x=0
        y=0
    return (x, y)



  
def main():
    
    last_x=0
    last_y=0

    
    while True:
          ret, image=cam.read()
          start=time.time()
          model = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb',
                                      'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
          cur_x, cur_y=detect(image)
          cv2.line(image, (cur_x, cur_y), (last_x, last_y), (0, 0, 200), 5)
          last_x=cur_x
          last_y=cur_y
          image_height, image_width, _ = image.shape
          model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
          output = model.forward()

          for detection in output[0, 0, :, :]:
                   confidence = detection[2]
                   if confidence > .5:
                            class_id = detection[1]
                            class_name=id_class_name(class_id,classNames)
                            print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
                            box_x = detection[3] * image_width
                            box_y = detection[4] * image_height
                            box_width = detection[5] * image_width
                            box_height = detection[6] * image_height
                            SpeedEstimatorTool=SpeedEstimator([int(box_x), int(box_y)],fps)
                            speed=SpeedEstimatorTool.estimateSpeed()
                            cv2.putText(image,class_name+": "+str(speed)+"Km/h",(int(box_x), int(box_y+.05*image_height)),cv2.FONT_HERSHEY_SIMPLEX,(.005*image_width),(0, 0, 255))
                            cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                            end=time.time()
                            print(f"{end-start:.15f} sec")
                            if last_x is 0 & last_y is 0:
                                print('Image load failed!')
                            else:
                                current_time=datetime.now()
                                ye=current_time.year
                                mo=current_time.month
                                da=current_time.day
                                ho=current_time.hour
                                mi=current_time.minute
                                se=current_time.second
                                cv2.imwrite("C:\\opencv\%d %d %d %d %d %d.jpg" %(ye, mo, da, ho, mi, se), image) 
                                print("saved image %d %d %d %d %d %d.jpg"  %(ye, mo, da, ho, mi, se))
       
          cv2.imshow('video', image)
          if cv2.waitKey(10) & 0xFF == ord('q'):
              break
          
    cam.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()

