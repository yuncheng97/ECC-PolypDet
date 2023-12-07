import os
import cv2
import math
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import argparse

class TrainData(Dataset):
    """
    A custom dataset class for training data.
    """
    def __init__(self, args):
        """
        Initialize the TrainData object with the given arguments.
        
        :param args: Arguments containing the data path and other configurations.
        """
        # Initialize variables
        self.args       = args
        self.in_size    = 512  # Input image size: 512, 256, 128, 64
        self.out_size   = self.in_size // 4
        # Define image transformations
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(self.in_size, self.in_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc'))

        self.samples = []

        # Read in training data
        with open(args.data_path + '/train_box.txt') as lines:
            for line in lines:
                name, boxs = line.strip().split(';')
                boxs = boxs.split(' ')
                bbox = []
                for i in range(len(boxs) // 4):
                    xmin, ymin, xmax, ymax = boxs[4 * i:4 * (i + 1)]
                    bbox.append([max(int(xmin), 0), max(int(ymin), 0), int(xmax), int(ymax), 0])
                self.samples.append([name, bbox])
        print('training samples:', len(self.samples))

    def __getitem__(self, idx):
        """
        Get the item at the given index.
        
        :param idx: Index of the item to be fetched.
        :return: Tuple containing image, heatmap, inter_heatmap, height_width, center_reg, center_reg_mask, and mask.
        """
        name, bbox          = self.samples[idx]
        # image = cv2.imread(self.args.data_path + '/train/' + name)
        name                = self.args.data_path + '/TrainDataset/Frame/' + name
        image               = cv2.imread(name)
        image               = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask                = np.zeros((image.shape[0], image.shape[1]))
        pair                = self.transform(image=image, mask=mask, bboxes=bbox)
        image, mask, bboxes = pair['image'], pair['mask'], np.array(pair['bboxes'])
        bboxes              = bboxes / 4
        heatmap             = np.zeros((self.out_size, self.out_size), dtype=np.float32)
        inter_heatmap       = np.zeros((self.out_size, self.out_size), dtype=np.float32)
        height_width        = np.zeros((self.out_size, self.out_size, 2), dtype=np.float32)
        center_reg          = np.zeros((self.out_size, self.out_size, 2), dtype=np.float32)
        center_reg_mask     = np.zeros((self.out_size, self.out_size), dtype=np.float32)
        
        # Loop through bounding boxes and create heatmaps
        for bbox in bboxes:
            xmin, ymin, xmax, ymax     = bbox[:4]
            mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
            h, w                       = ymax - ymin, xmax - xmin
            radius                     = gaussian_radius(math.ceil(h), math.ceil(w))
            # Calculate the feature point that the true box belongs to
            cx, cy                     = (xmin + xmax) / 2, (ymin + ymax) / 2
            cxi, cyi                   = int(cx), int(cy)
            # Draw Gaussian heatmap
            heatmap                    = draw_gaussian(heatmap, (cxi, cyi), radius)
            inter_heatmap              = draw_gaussian(inter_heatmap, (cxi, cyi), 2 * radius)
            # Calculate true width and height values
            height_width[cyi, cxi]     = w, h
            # Calculate center offset
            center_reg[cyi, cxi]       = cx - cxi, cy - cyi
            # Set corresponding mask to 1
            center_reg_mask[cyi, cxi]  = 1
        
        heatmap         = torch.from_numpy(heatmap)
        inter_heatmap   = torch.from_numpy(inter_heatmap)
        height_width    = torch.from_numpy(height_width)
        center_reg      = torch.from_numpy(center_reg)
        center_reg_mask = torch.from_numpy(center_reg_mask)
        return image, heatmap, inter_heatmap, height_width, center_reg, center_reg_mask, mask

    def __len__(self):
        """
        Get the length of the dataset.
        
        :return: Length of the dataset.
        """
        return len(self.samples)


class ReTrainData(Dataset):
    def __init__(self, args):
        self.args      = args
        self.in_size   = 512
        self.out_size  = self.in_size//4

        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(self.in_size, self.in_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc'))

        self.samples   = []

        with open(args.data_path+'/retrain_list.txt') as lines:
            for line in lines:
                name, boxs, label = line.strip().split(';')
                boxs       = boxs.split(' ')
                bbox       = []
                for i in range(len(boxs)//4):
                    xmin, ymin, xmax, ymax = boxs[4*i:4*(i+1)]
                    bbox.append([max(int(xmin),0), max(int(ymin),0), int(xmax), int(ymax), 0])
                self.samples.append([name, bbox, label])
    
        print('traning samples:', len(self.samples))

    def __getitem__(self, idx):
        name, bbox, label   = self.samples[idx]
        # image               = cv2.imread(self.args.data_path+'/train/'+name)
        image               = cv2.imread(self.args.data_path+'/TrainDataset/Frame/'+name)
        image               = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask                = np.zeros((image.shape[0], image.shape[1]))
        pair                = self.transform(image=image, mask=mask, bboxes=bbox)
        image, mask, bboxes = pair['image'], pair['mask'], np.array(pair['bboxes'])
        bboxes              = bboxes/4
        heatmap             = np.zeros((self.out_size, self.out_size   ), dtype=np.float32)
        inter_heatmap       = np.zeros((self.out_size, self.out_size   ), dtype=np.float32)
        height_width        = np.zeros((self.out_size, self.out_size, 2), dtype=np.float32)
        center_reg          = np.zeros((self.out_size, self.out_size, 2), dtype=np.float32)
        center_reg_mask     = np.zeros((self.out_size, self.out_size   ), dtype=np.float32)
        
        weight              = np.array([float(label)], dtype=np.float32)
        for bbox in bboxes:
            xmin, ymin, xmax, ymax    = bbox[:4]
            mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
            h, w                      = ymax-ymin, xmax-xmin
            radius                    = gaussian_radius(math.ceil(h), math.ceil(w))
            # 计算真实框所属的特征点
            cx, cy                    = (xmin+xmax)/2, (ymin+ymax)/2
            cxi, cyi                  = int(cx), int(cy)
            # 绘制高斯热力图
            heatmap                   = draw_gaussian(heatmap, (cxi, cyi), radius)
            inter_heatmap             = draw_gaussian(inter_heatmap, (cxi, cyi), 2 * radius)
            # 计算宽高真实值
            height_width[cyi, cxi]    = w, h
            # 计算中心偏移量
            center_reg[cyi, cxi]      = cx-cxi, cy-cyi
            # 将对应的mask设置为1
            center_reg_mask[cyi, cxi] = 1
        
        heatmap         = torch.from_numpy(heatmap)
        inter_heatmap   = torch.from_numpy(inter_heatmap)
        height_width    = torch.from_numpy(height_width)
        center_reg      = torch.from_numpy(center_reg)
        center_reg_mask = torch.from_numpy(center_reg_mask)
        weight          = torch.from_numpy(weight)
        return image, heatmap, inter_heatmap, height_width, center_reg, center_reg_mask, mask, weight

    def __len__(self):
        return len(self.samples)
    


def draw_gaussian(heatmap, center, radius, k=1):
    diameter        = 2*radius + 1
    gaussian        = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y            = int(center[0]), int(center[1])
    height, width   = heatmap.shape[0:2]
    left, right     = min(x, radius), min(width-x, radius+1)
    top, bottom     = min(y, radius), min(height-y, radius+1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(height, width , min_overlap=0.7):
    a1  = 1
    b1  = (height+width)
    c1  = width*height*(1-min_overlap) / (1+min_overlap)
    sq1 = np.sqrt(b1**2 - 4*a1*c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2*(height+width)
    c2  = (1-min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2  = (b2 + sq2)/2

    a3  = 4*min_overlap
    b3  = -2*min_overlap*(height + width)
    c3  = (min_overlap - 1)*width*height
    sq3 = np.sqrt(b3**2 - 4*a3*c3)
    r3  = (b3 + sq3) / 2
    return int(min(r1, r2, r3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone'   , type=str   , default='pvt_v2_b2'              )
    parser.add_argument('--snapshot'   , type=str   , default=None                     )
    parser.add_argument('--epoch'      , type=int   , default=20                       )
    parser.add_argument('--time_clips' , type=int   , default=2                        )
    parser.add_argument('--lr'         , type=float , default=1e-4                     )
    parser.add_argument('--batch_size' , type=int   , default=16                       )
    parser.add_argument('--data_path'  , type=str   , default='/mntnfs/med_data5/yuncheng/DATASET/ZSPolyp'            )
    parser.add_argument('--model_path' , type=str   , default='/mntnfs/med_data5/yuncheng/centernet/centernet/model/test'               )
    args = parser.parse_args()

    dataset = TrainData(args)
    # for i in range(50):
    image, heatmap, inter_heatmap, height_width, center_reg, center_reg_mask, mask = dataset.__getitem__(0)
