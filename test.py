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
import logging
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

from model import PolypModel, CascadePolypModel, ContrastivePolypModel
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
        with open(args.data_path+'/Image_List'+args.file_name) as lines:
            for line in lines:
                name, boxs = line.strip().split(';')
                boxs       = boxs.split(' ')
                bbox       = []
                for i in range(len(boxs)//4):
                    xmin, ymin, xmax, ymax = boxs[4*i:4*(i+1)]
                    bbox.append([max(int(xmin),0), max(int(ymin),0), int(xmax), int(ymax)])
                self.samples.append([name, bbox])
        print('test samples:', len(self.samples))

        ###### inference all video clips
        # for folder in os.listdir(args.data_path+'/TestEasyDataset/Frame'):
        #     # if folder == '.DS_Store':
        #     #     continue
        #     for name in os.listdir(args.data_path+'/TestEasyDataset/Frame/'+folder):
        #         if name.endswith('.jpg'):
        #             # self.names.append(folder+'/'+name[0:4])
        #             self.names.append(folder+'/'+name.split('.')[0])

        # print(self.names)
        # print('test samples:', len(self.names))

        ##### inference some video clips
        # data_path = args.data_path+'/test'
        # for folder in sorted(os.listdir(data_path), key=lambda x:int(x)):
        #     if int(folder) in range(1, 2):
        #         for name in os.listdir(data_path+'/'+folder):
        #             self.names.append(folder+'/'+name[0:4])
        # print('test samples:', len(self.names))

        #### inference single video clip
        # for name in os.listdir('/mntnfs/med_data5/yuncheng/DATASET/SUN-SEG/TestHardDataset/Frame/case42'):
        # # for name in os.listdir('/mntnfs/med_data5/yuncheng/DATASET/SCHPolyp/test/199'):
        #     if name.endswith('.jpg'):
        #         self.names.append('case42/'+name[0:-4])
        #         self.names.append('32/'+name[0:-4])
        
        ## model
        if args.model_name == 'PolypModel':
            self.model = PolypModel(args).cuda()
        elif args.model_name == 'ContrastivePolypModel':
            self.model = ContrastivePolypModel(args).cuda()
        self.model.eval()
    
    def show(self):
        with torch.no_grad():
            np.random.shuffle(self.names)
            for idx, name in enumerate(self.names):
                image  = cv2.imread('/mntnfs/med_data5/yuncheng/DATASET/ZSPolyp/train/1/0001.jpg')
                mask   = cv2.imread('/mntnfs/med_data5/yuncheng/DATASET/ZSPolyp/train/1/0001.png')
                image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                H,W,C  = image.shape
                origin = image

                image  = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)/255.0
                image  = (image-self.mean)/self.std
                image  = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).cuda().float()
                heatmap, whpred, offset = self.model(image)
                outputs = decode_bbox(heatmap, whpred, offset, self.confidence)
                results = postprocess(outputs, (H,W), self.nms_iou)
                if results[0] is None:
                    continue
                confidence = results[0][:, 4]
                bboxes     = results[0][:, :4]

                for box in bboxes:
                    ymin, xmin, ymax, xmax = box
                    origin = cv2.rectangle(origin, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0,255,0), thickness=5)
                plt.subplot(121)
                plt.imshow(origin)
                plt.subplot(122)
                plt.imshow(np.uint8(origin*0.5+mask*0.5))
                plt.savefig(str(idx)+'.png')
                print('saved figure!')
                quit()
                plt.cla()
    

    def accuracy(self):
        with torch.no_grad():
            start = time.time()
            num_tps, num_fps, num_dets, num_gts = 0, 0, 0, 0
            for idx in tqdm(range(len(self.samples))):
                name, bbox = self.samples[idx]
                image    = cv2.imread(self.args.data_path+'/TestDataset/'+self.args.test_data+'/images/'+name)
                # image    = cv2.imread(self.args.data_path+'/TestDataset/CVC-ClinicDB/images/'+name)
                # image    = cv2.imread(self.args.data_path+'TestEasyDataset/Frame/'+name)
                # image    = cv2.imread(self.args.data_path+'/images/'+name)
                # image    = cv2.imread(self.args.data_path+'/Original/'+name)
                # image    = cv2.imread(self.args.data_path+'/test/Image/'+name)


                image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                H,W,C    = image.shape

                gt_bboxs = bbox
                image      = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)/255.0
                image      = (image-self.mean)/self.std
                image      = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).cuda().float()
                if self.args.model_name == 'PolypModel':
                    heatmap, whpred, offset = self.model(image)
                elif self.args.model_name == 'ContrastivePolypModel':
                    _, _, heatmap, whpred, offset = self.model(image)
                outputs    = decode_bbox(heatmap, whpred, offset, self.confidence)
                results    = postprocess(outputs, (H,W), self.nms_iou)
                if results[0] is None:
                    num_gts += len(gt_bboxs)
                    continue
                confidence = results[0][:, 4]
                bboxs      = []
                for box in results[0][:, :4]:
                    ymin, xmin, ymax, xmax = box
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    bboxs.append([xmin, ymin, xmax, ymax])
                num_tp, num_fp, num_det, num_gt = num_iou(bboxs, gt_bboxs)
                num_tps, num_fps, num_dets, num_gts = num_tps+num_tp, num_fps+num_fp, num_dets+num_det, num_gts+num_gt 
        end = time.time()
        seconds = end - start
        print ("Time taken : {:0.1f} seconds".format(seconds))
        fps  = len(self.samples) / seconds
        print("Estimated frames per second : {:0.1f}".format(fps))
        print('precision=%f, recall=%f, f1=%f, f2=%f'%(num_tps/num_dets, num_tps/num_gts, 2*num_tps/(num_dets+num_gts), (5*num_tps) / (4*num_gts + num_dets)))
        with open("/home/yunchengjiang/centernet/logs/centernet/results/sample_b20_e200_s352_logs.txt", 'a') as file:
            precision = round(num_tps/num_dets, 3)
            recall    = round(num_tps/num_gts, 3)
            f1        = round(2*num_tps/(num_dets+num_gts), 3)
            f2        = round((5*num_tps) / (4*num_gts + num_dets), 3)
            line = self.args.test_data+' precision='+str(precision)+' recall='+str(recall)+' f1='+str(f1)+' f2='+str(f2)+'\n'
            file.write(line)
            print("Finish testing ", self.args.test_data)

    def inference(self):
        with torch.no_grad():
            for name in tqdm(self.names):
                image  = cv2.imread(self.args.data_path+'/TestHardDataset/Frame/'+name+'.jpg')
                # image  = cv2.imread(self.args.data_path+'/test/'+name+'.jpg')
                origin = image
                mask   = cv2.imread(self.args.data_path+'/TestHardDataset/GT/'+name+'.png')
                # mask  = cv2.imread(self.args.data_path+'/test/'+name+'.png')
                image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                H,W,C  = image.shape

                image  = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)/255.0
                image  = (image-self.mean)/self.std
                image  = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).cuda().float()
                _, _, heatmap, whpred, offset = self.model(image)
                outputs = decode_bbox(heatmap, whpred, offset, self.confidence)
                results = postprocess(outputs, (H,W), self.nms_iou)
                if results[0] is None:
                    continue
                confidence = results[0][:, 4]
                bboxes     = results[0][:, :4]
                for box in bboxes:
                    ymin, xmin, ymax, xmax = box

                    # text_size = cv2.getTextSize('Polyp:%.2f' % confidence[0], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    # cv2.rectangle(origin, (int(xmin), int(ymin)), (int(xmin)+text_size[0], int(ymin)+text_size[1]), (1,142,35), -1)
                    origin = cv2.rectangle(origin, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0,255,0), thickness=8)
                    # cv2.putText(origin, 'Polyp:%.2f' % confidence[0], (int(xmin), int(ymin)+text_size[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                figure_save_folder = '/mntnfs/med_data5/yuncheng/DATASET/SUN-SEG/Inference/' + name.split('/')[0]
                # figure_save_folder = '/mntnfs/med_data5/yuncheng/DATASET/SCHPolyp/Inference/centernet_cascade_contrastive/' + name.split('/')[0]
                figure_save_name = name.split('/')[-1]+'.png'
                if not os.path.exists(figure_save_folder):
                    os.makedirs(figure_save_folder) # 如果不存在目录figure_save_path，则创建
                # cv2.imwrite(os.path.join(figure_save_folder, figure_save_name), np.uint8(origin*0.5+mask*0.5))
                cv2.imwrite(os.path.join(figure_save_folder, figure_save_name), np.uint8(origin))
    
    def create_gif(self, path):
        frames = []
        for name in tqdm(sorted(os.listdir(path))):
            image = imageio.imread(path+'/'+name)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
        imageio.mimsave(path+'.mp4', frames, 'MP4')
        return

    def save_gt(self, read_path, save_path):
        image = cv2.imread(read_path)
        mask = cv2.imread(read_path.replace('.jpg', '.png'), cv2.IMREAD_GRAYSCALE)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        gt_bboxs = []
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            gt_bboxs.append([x, y, x+w, y+h])
        for box in gt_bboxs:
            xmin, ymin, xmax, ymax = box
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0,255,0), thickness=8)
        
        cv2.imwrite(save_path, np.uint8(image))
        print("gt saved")
    
    def save_mm_prediction(self, read_path, save_path):
        image = cv2.imread(read_path)
        gt_bboxs = [[1022,811,1149,949]]
        for box in gt_bboxs:
            xmin, ymin, xmax, ymax = box
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0,255,0), thickness=8)
        
        cv2.imwrite(save_path, np.uint8(image))
        print("mm prediction saved")




def is_valid_bbox(args):
    mean       = np.array([0.485, 0.456, 0.406])
    std        = np.array([0.229, 0.224, 0.225])
    confidence = 0.3
    nms_iou    = 0.3
    model = ContrastivePolypModel(args).cuda()
    model.eval()
    path = '/mntnfs/med_data5/yuncheng/DATASET/test-data-with-invalid-bbox'
    for item in os.listdir(path):
        name = path + '/' + item
        print(name)
        with torch.no_grad():
            image    = cv2.imread(name)
            image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            H,W,C    = image.shape
            image    = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)/255.0
            image    = (image-mean)/std
            image    = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).cuda().float()
            # _, heatmap, whpred, offset = self.model(image)
            # _, heatmap, whpred, offset = self.model(image)
            _, _, heatmap, whpred, offset = model(image)
            outputs    = decode_bbox(heatmap, whpred, offset, confidence)
            results    = postprocess(outputs, (H,W), nms_iou)
            if results[0] is None:
                continue
            for box in results[0][:, :4]:
                ymin, xmin, ymax, xmax = box
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

                if ( xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0 or xmax > H or ymax > W):
                    print(f"invalid image:{item}, xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}")
                if xmin < xmax and ymin < ymax:
                    # print(f"xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}")
                    continue
                else:
                    print(f"invalid image:{item}, xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}")
                    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot'  , type=str, default=None    )
    parser.add_argument('--data_path' , type=str, default=None    )
    parser.add_argument('--test_data' , type=str, default=None    )
    parser.add_argument('--file_name' , type=str, default=None    )
    parser.add_argument('--backbone'  , type=str, default=None    )
    parser.add_argument('--model_name', type=str, default=None    )
    # parser.add_argument('--snapshot'  , default='/mntnfs/med_data5/yuncheng/centernet/centernet/model/centernet_sun_cascade_contrast_hard/best.pth'    )
    args = parser.parse_args()

    print('Testing Model:', args.model_name)
    print('Testing Backbone:', args.backbone)
    print("Testing Dataset:", args.data_path)
    t = Test(args)

    t.accuracy()
    # t.show()
    # t.test()
    # t.inference()
    # t.create_gif('/mntnfs/med_data5/yuncheng/DATASET/SUN-SEG/Inference/case42')
