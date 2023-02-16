import os
import cv2
import matplotlib.pyplot as plt
path = '/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp/train/5/0001.jpg'
image = cv2.imread(path)
mask = cv2.imread(path.replace('.jpg', '.png'))
contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
gt_bboxs = []
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    gt_bboxs.append([x, y, x+w, y+h])

for box in gt_bboxs:
    ymin, xmin, ymax, xmax = box
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)