# -*- coding: UTF-8 -*-
import os
from posixpath import ismount
import cv2
import sys
import torch
import time
import argparse
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn import manifold
import time
import logging
from tabulate import tabulate
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

from model import CenterNet, ECCPolypDet
from utils import decode_bbox, postprocess

def num_iou(bboxs, gt_bboxs):
    num_tp = 0
    for box in bboxs:
        flag = False
        for gt_box in gt_bboxs:
            xmin, ymin, xmax, ymax = box
            x1, y1, x2, y2         = gt_box
            width, height          = max(min(xmax, x2)-max(xmin, x1), 0), max(min(ymax, y2)-max(ymin, y1), 0)
            union                  = (xmax-xmin)*(ymax-ymin)+(x2-x1)*(y2-y1)
            inter                  = width*height
            iou                    = inter/(union-inter)
            if iou>0.5 and width>0 and height>0:       
                flag = True
                break
        if flag:
            num_tp += 1
    return num_tp, len(bboxs)-num_tp, len(bboxs), len(gt_bboxs)


class Test:
    def __init__(self, args):
        self.args       = args
        self.mean       = np.array([0.485, 0.456, 0.406])
        self.std        = np.array([0.229, 0.224, 0.225])
        self.confidence = 0.3
        self.nms_iou    = 0.3
        ## data
        self.names      = []
        self.samples    = []

        # Choose the mode of operation: 'txtfile', 'all_video_clips', 'some_video_clips', or "single_video_clip"
        mode = 'txtfile'

        if mode == 'txtfile':
            # inference by txtfile
            with open(args.data_path+'/'+args.file_name) as lines:
                for line in lines:
                    name, boxs = line.strip().split(';')
                    boxs       = boxs.split(' ')
                    bbox       = []
                    for i in range(len(boxs)//4):
                        xmin, ymin, xmax, ymax = boxs[4*i:4*(i+1)]
                        bbox.append([max(int(xmin),0), max(int(ymin),0), int(xmax), int(ymax)])
                    self.samples.append([name, bbox])
            print('test samples:', len(self.samples))
        elif mode == 'all_video_clips':
            ##### inference all video clips
            for folder in os.listdir(args.data_path+'/TestEasyDataset/Frame'):
                for name in os.listdir(args.data_path+'/TestEasyDataset/Frame/'+folder):
                    if name.endswith('.jpg'):
                        self.samples.append(folder+'/'+name.split('.')[0])

            print(self.samples)
            print('test samples:', len(self.samples))

        elif mode == 'some_video_clips':
            #### inference some video clips
            data_path = args.data_path+'/test'
            for folder in sorted(os.listdir(data_path), key=lambda x:int(x)):
                if int(folder) in range(1, 2):
                    for name in os.listdir(data_path+'/'+folder):
                        self.samples.append(folder+'/'+name[0:4])
            print('test samples:', len(self.samples))

        else:
            #### inference single video clip
            for name in os.listdir('/220019054/Dataset/SUN-SEG/TrainDataset/Frame/case2_7'):
                if 'mask' not in name:
                    self.samples.append('case2_7/'+name[0:-4])
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
    def accuracy(self):
        print('start testing accuracy ...')
        with torch.no_grad():
            start = time.time()
            num_tps, num_fps, num_dets, num_gts = 0, 0, 0, 0
            for idx in tqdm(range(len(self.samples))):
                name, bbox = self.samples[idx]
                image    = cv2.imread(self.args.data_path+'TestEasyDataset/Frame/'+name)
                image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                H,W,C    = image.shape
                gt_bbox = bbox
                image      = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)/255.0
                image      = (image-self.mean)/self.std
                image      = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).cuda().float()
                if self.args.model_name == 'CenterNet':
                    heatmap, whpred, offset = self.model(image)
                elif self.args.model_name == 'ECCPolypDet':
                    _, _, _, heatmap, whpred, offset = self.model(image)
                outputs    = decode_bbox(heatmap, whpred, offset, self.confidence)
                results    = postprocess(outputs, (H,W), self.nms_iou)
                if results[0] is None:
                    num_gts += len(gt_bbox)
                    continue
                confidences = results[0][:, 4]
                bboxes      = results[0][:, :4]
                pred_bbox         = []
                for bbox, confidence in zip(bboxes, confidences):
                    if confidence >= 0.5:
                        ymin, xmin, ymax, xmax = bbox
                        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                        pred_bbox.append([xmin, ymin, xmax, ymax])
                num_tp, num_fp, num_det, num_gt = num_iou(pred_bbox, gt_bbox)
                num_tps, num_fps, num_dets, num_gts = num_tps+num_tp, num_fps+num_fp, num_dets+num_det, num_gts+num_gt 
        end = time.time()
        seconds = end - start
        print ("Time taken : {:0.1f} seconds".format(seconds))
        fps  = len(self.samples) / seconds
        print("Estimated frames per second : {:0.1f}".format(fps))
        print('precision=%f, recall=%f, f1=%f, f2=%f'%(num_tps/(num_dets+1e-6), num_tps/num_gts, 2*num_tps/(num_dets+num_gts), (5*num_tps) / (4*num_gts + num_dets)))


    def t_sne(self):
        print('start generating t-sne map ...')
        with torch.no_grad():
            img_size   = 64
            hidden_dim = 64
            name       = 'case2_7/case_M_20181003094031_0U62363100354631_1_001_002-1_a13_ayy_image0001'
            image      = cv2.imread(self.args.data_path+'/TrainDataset/Frame/'+name+'.jpg')
            origin     = image.copy()
            image      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask       = np.float32(cv2.imread(self.args.data_path+'/TrainDataset/GT/'+name+'.png', cv2.IMREAD_GRAYSCALE)>128)
            mask       = cv2.resize(mask, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
            mask       = np.where(mask>0, 1.0, 0.0)
            image      = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR) / 255.0
            image      = (image - self.mean) / self.std
            image      = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda().float()
            pred, _, _, heatmap, whpred, offset = self.model(image)
            feature    = pred.squeeze(0).cpu().detach().numpy()
            feature    = np.transpose(feature, (1,2,0))
            feature    = cv2.resize(feature, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
            feature    = np.reshape(feature, (-1, feature.shape[2]))   #[64*64, 64]
            
            features   = feature
            labels     = mask.flatten()

            tsne         = manifold.TSNE(n_components=2, init='pca', random_state=42)
            start_time   = time.time()
            tsne_2d         = tsne.fit_transform(features)
            
            # draw origin image t-sne
            # image_1d     = cv2.resize(origin, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
            # image_1d     = np.reshape(image_1d, (-1, 3))
            # tsne_2d      = tsne.fit_transform(image_1d)
            # #
            
            x_min, x_max = tsne_2d.min(0), tsne_2d.max(0)
            tsne_norm    = (tsne_2d - x_min) / (x_max - x_min)
            elipse       = time.time()-start_time
            print('cost time:', elipse)

            back_idxs = (labels == 0.0)
            fore_idxs = (labels == 1.0)
            tsne_back = tsne_norm[back_idxs]
            tsne_fore = tsne_norm[fore_idxs]
            fig       = plt.figure(figsize=(12, 12))
            plt.scatter(tsne_back[:, 0], tsne_back[:, 1], 1, color='red', label='background pixel')
            plt.scatter(tsne_fore[:, 0], tsne_fore[:, 1], 1, color='green', label='foreground pixel')
            plt.legend(loc='upper left')
            plt.show()
            plt.savefig('t-sne_w_bacl.png', dpi=300)
        return 

    def inference(self):
        print("start inference ...")
        with torch.no_grad():
            for name, bbox in tqdm(self.samples):
                image  = cv2.imread(self.args.data_path+'/TestHardDataset/Frame/'+name)
                origin = image.copy()
                image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                H, W, C = image.shape
                image  = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR) / 255.0
                image  = (image - self.mean) / self.std
                image  = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda().float()
                _, _, _, heatmap, whpred, offset = self.model(image)
                outputs = decode_bbox(heatmap, whpred, offset, self.confidence)
                results = postprocess(outputs, (H, W), self.nms_iou)
                if results[0] is None:
                    print("can't detect anything!")
                    continue
                confidences = results[0][:, 4]
                bboxes      = results[0][:, :4]
                for box, confidence in zip(bboxes, confidences):
                    if confidence >= 0:
                        ymin, xmin, ymax, xmax = box
                        text_size = cv2.getTextSize('Polyp:%.2f' % confidence, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                        cv2.rectangle(origin, (int(xmin), int(ymin)), (int(xmin)+text_size[0], int(ymin)-2*text_size[1]), (1,142,35), -1)
                        origin = cv2.rectangle(origin, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0,255,0), thickness=5)
                        cv2.putText(origin, 'Polyp:%.2f' % confidence, (int(xmin), int(ymin)-text_size[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                figure_save_folder = '/220019054/JBHI23-ECCPolypDet/inference/' + name.split('/')[0]

                figure_save_name = name.split('/')[-1]
                os.makedirs(figure_save_folder, exist_ok=True)  # If the directory figure_save_path does not exist, create it
                cv2.imwrite(os.path.join(figure_save_folder, figure_save_name), np.uint8(origin))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path' , type=str, default='/220019054/Dataset/SUN-SEG'    )
    parser.add_argument('--task' , type=str, default=None    )
    parser.add_argument('--file_name' , type=str, default=None    )
    parser.add_argument('--backbone'  , type=str, default='pvt_v2_b2'    )
    parser.add_argument('--model_name', type=str, default='ECCPolypDet'    )
    parser.add_argument('--pretrained'  , default=None    )
    args = parser.parse_args()

    print('task:', args.task)
    print('model name:', args.model_name)
    print('pretrained:', args.pretrained)
    tester = Test(args)

    if args.task == 'inference':
        tester.inference()
    elif args.task == 'accuracy':
        tester.accuracy()
    else:
        tester.t_sne()
