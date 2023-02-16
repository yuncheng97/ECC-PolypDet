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
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

from model import PolypModel, CascadePolypModel, ContrastivePolypModel, Cascade2PolypModel
from utils import decode_bbox, postprocess


def num_iou(bboxs, gt_bboxs):
    num_tp = 0
    flag = True
    IOU = []
    for box in bboxs:
        for gt_box in gt_bboxs:
            xmin, ymin, xmax, ymax = box
            x1, y1, x2, y2         = gt_box
            width, height          = min(xmax, x2)-max(xmin, x1), min(ymax, y2)-max(ymin, y1)
            union                  = (xmax-xmin)*(ymax-ymin)+(x2-x1)*(y2-y1)
            inter                  = width*height
            iou                    = inter/(union-inter)
            IOU.append(iou)
            if iou>0.9 and width>0 and height>0:       
                flag = False
                break
        if flag == False:
            break
    return flag, IOU
    

'''
    所有预测bbox与真实bbox的IOU都小于0.x即认为是难样本
'''
class HardMinging:
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
        # self.model = PolypModel(args).cuda()
        self.model  = CascadePolypModel(args).cuda()
        # self.model = ContrastivePolypModel(args).cuda()
        # self.model   = Cascade2PolypModel(args).cuda()
        self.model.eval()
        

    def inference(self):
        with torch.no_grad():
            with open('/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp/hard_cas_train_box.txt', 'w') as hard_samples:
                for idx in range(len(self.samples)):
                    name, bbox = self.samples[idx]
                    image    = cv2.imread(self.args.data_path+'/train/'+name)
                    mask     = cv2.imread(self.args.data_path+'/train/'+name.split('.')[0]+'.png')
                    image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    H,W,C    = image.shape
                    origin = image

                    gt_bboxs = bbox
                    image      = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)/255.0
                    image      = (image-self.mean)/self.std
                    image      = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).cuda().float()
                    # heatmap, whpred, offset = self.model(image)
                    _, heatmap, whpred, offset = self.model(image)
                    # _, _, heatmap, whpred, offset = self.model(image)
                    outputs    = decode_bbox(heatmap, whpred, offset, self.confidence)
                    results    = postprocess(outputs, (H,W), self.nms_iou)
                    # 注意这里，如果没有检测到物体，也判断该图片为难样本
                    if results[0] is None:
                        hard_samples.write(self.names[idx][:-1]+';'+'hard'+'\n')
                        continue
                    confidence = results[0][:, 4]
                    bboxs      = []
                    for box in results[0][:, :4]:
                        ymin, xmin, ymax, xmax = box
                        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                        bboxs.append([xmin, ymin, xmax, ymax])
                        origin = cv2.rectangle(origin, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0,255,0), thickness=5)
                    flag, IOU = num_iou(bboxs, gt_bboxs)
                    # 如果是难样本，则记录图片编号
                    if flag:
                        print(name, 'hard sample!', IOU)
                        hard_samples.write(self.names[idx][:-1]+';'+'hard'+'\n')
                    else:
                        print(name, 'easy sample!', IOU)
                        hard_samples.write(self.names[idx][:-1]+';'+'easy'+'\n')
                    # figure_save_folder = '/mntnfs/med_data4/yuncheng/DATASET/ZSPolyp/inference/train/' + name.split('/')[0]
                    # figure_save_name = name.split('/')[-1]+'.png'
                    # if not os.path.exists(figure_save_folder):
                    #     os.makedirs(figure_save_folder) # 如果不存在目录figure_save_path，则创建
                    # cv2.imwrite(os.path.join(figure_save_folder, figure_save_name), np.uint8(origin*0.5+mask*0.5))






if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot'  , default='/mntnfs/med_data4/yuncheng/centernet/centernet/model/centernet_sch_cascade/19.pth'    )
    parser.add_argument('--data_path' , default='/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp'       )
    parser.add_argument('--backbone'  , default='pvt_v2_b2'         )
    args = parser.parse_args()

    t = HardMinging(args)
    t.inference()
