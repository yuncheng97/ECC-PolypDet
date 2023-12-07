import os
from posixpath import ismount
import cv2
import sys
import torch
import argparse
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

from model import CenterNet, ECCPolypDet
from utils import decode_bbox, postprocess


def num_iou(bboxs, gt_bboxs):
    num_tp = 0
    flag = True
    IOU = []
    for box in bboxs:
        for gt_box in gt_bboxs:
            xmin, ymin, xmax, ymax = box
            x1, y1, x2, y2         = gt_box
            width, height          = max(min(xmax, x2)-max(xmin, x1), 0), max(min(ymax, y2)-max(ymin, y1), 0)
            union                  = (xmax-xmin)*(ymax-ymin)+(x2-x1)*(y2-y1)
            inter                  = width*height
            iou                    = inter/(union-inter)
            IOU.append(iou)
            if iou>0.7 and width>0 and height>0:       
                flag = False
    return flag, IOU
    

'''
    所有预测bbox与真实bbox的IOU都小于0.x即认为是难样本
'''
class AdaptiveHardMinging:
    def __init__(self, args):
        self.args       = args
        self.mean       = np.array([0.485, 0.456, 0.406])
        self.std        = np.array([0.229, 0.224, 0.225])
        self.confidence = 0.3
        self.nms_iou    = 0.3
        ## data
        self.names      = []
        self.samples    = []
        with open(args.data_path+'/train_box.txt') as lines:
            for line in lines:
                name, boxs = line.strip().split(';')
                boxs       = boxs.split(' ')
                bbox       = []
                for i in range(len(boxs)//4):
                    xmin, ymin, xmax, ymax = boxs[4*i:4*(i+1)]
                    bbox.append([max(int(xmin),0), max(int(ymin),0), int(xmax), int(ymax)])
                self.samples.append([name, bbox])
                self.names.append(line)
        print('test samples:', len(self.samples))
        
        ## model
        model_dict = {
            'CenterNet': CenterNet,
            'ECCPolypDet': ECCPolypDet,
        }

        if args.model_name in model_dict:
            self.model = model_dict[args.model_name](args).cuda()
        else:
            print(f"Invalid model name: {args.model_name}")
        self.model.eval()
        

    def inference(self):
        print('start hard sample mining ...')
        max_weight = 1
        with torch.no_grad():
            with open('/220019054/JBHI23-ECCPolypDet/retrain/retrain_list.txt', 'w') as hard_samples:
                for idx in tqdm(range(len(self.samples))):
                    name, bbox = self.samples[idx]
                    image      = cv2.imread(self.args.data_path+'/TrainDataset/Frame/'+name)
                    mask       = cv2.imread(self.args.data_path+'/TrainDataset/GT/'+name.split('.')[0]+'.png')
                    image      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    H,W,C      = image.shape
                    origin     = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    gt_bboxs   = bbox
                    image      = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)/255.0
                    image      = (image-self.mean)/self.std
                    image      = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).cuda().float()
                    _, _, _, heatmap, whpred, offset = self.model(image)
                    outputs    = decode_bbox(heatmap, whpred, offset, self.confidence)
                    results    = postprocess(outputs, (H,W), self.nms_iou)
                    # 注意这里，如果没有检测到物体，也判断该图片为难样本
                    if results[0] is None:
                        print(name, 'hard sample!', [0])
                        hard_samples.write(self.names[idx][:-1]+';'+str(max_weight)+'\n')
                        continue
                    confidence = results[0][:, 4]
                    bboxs      = []
                    for box in results[0][:, :4]:
                        ymin, xmin, ymax, xmax = box
                        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                        bboxs.append([xmin, ymin, xmax, ymax])
                        origin = cv2.rectangle(origin, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0,255,0), thickness=5)
                    flag, IOU  = num_iou(bboxs, gt_bboxs)
                    mean_iou   = sum(IOU) / len(IOU)
                    weight     = 1-mean_iou
                    # 如果是难样本，则记录图片编号
                    print(f'{name}, mean iou:{mean_iou:0.4f}, iou:{IOU}, weight:{weight}')
                    hard_samples.write(self.names[idx][:-1]+';'+str(weight)+'\n')




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained'  , default=None                             )
    parser.add_argument('--data_path'   , default='/220019054/Dataset/SUN-SEG'       )
    parser.add_argument('--backbone'    , default='pvt_v2_b2'                        )
    parser.add_argument('--model_name'  , default='ECCPolypDet'                       )
    args = parser.parse_args()

    miner = AdaptiveHardMinging(args)
    miner.inference()
