#coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from videomodel import PolypModel, PolypModelV2, PolypModelV3

def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    image  = cv2.imread(img_path)
    image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12.8, 9.6))
    plt.subplot(151) #ax1对应的是2行2列中的第1个grid子图  
    plt.imshow(image.permute(2,1,0)) 
    # 获取模型输出的feature/score
    model.eval()
    heatmap, whpred, offset = self.model(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone'   , type=str   , default='pvt_v2_b2'              )
    parser.add_argument('--snapshot'   , type=str   , default=None                     )
    parser.add_argument('--epoch'      , type=int   , default=20                       )
    parser.add_argument('--time_clips' , type=int   , default=2                        )
    parser.add_argument('--alpha'      , type=int   , default=0.5                      )
    parser.add_argument('--lr'         , type=float , default=1e-4                     )
    parser.add_argument('--batch_size' , type=int   , default=16                       )
    parser.add_argument('--data_path'  , type=str   , default='../dataset/'            )
    parser.add_argument('--model_path' , type=str   , default='./model/'               )
    args = parser.parse_args()

    model       = PolypModel(args).cuda()
    img_path    = '/mntnfs/med_data4/yuncheng/DATASET/ZSPolyp/test/29/0001.jpg'
    save_path   = './CAM.jpg'
    draw_CAM(model, img_path, save_path)