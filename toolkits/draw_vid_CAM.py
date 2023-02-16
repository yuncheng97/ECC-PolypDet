#coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from videomodel import PolypModel, PolypModelV2, PolypModelV3
import argparse

def draw_CAM(model, img_path1, img_path2, save_path, transform=None, visual_heatmap=False):
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
    mean        = np.array([0.485, 0.456, 0.406])
    std         = np.array([0.229, 0.224, 0.225])

    image1      = cv2.imread(img_path1)
    image1      = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2      = cv2.imread(img_path2)
    image2      = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10.8, 5.6))
    # plt.subplot(151)  
    plt.subplot(161)  
    plt.imshow(cv2.resize(image1, (512, 512), interpolation=cv2.INTER_LINEAR)) 
    plt.axis('off')
    plt.title('support image', size=10)

    # plt.subplot(152)  
    plt.subplot(162)  
    plt.imshow(cv2.resize(image2, (512, 512), interpolation=cv2.INTER_LINEAR)) 
    plt.axis('off')
    plt.title('current image', size=10)

    image1      = cv2.resize(image1, (512, 512), interpolation=cv2.INTER_LINEAR)/255.0
    image1      = (image1-mean) / std

    image2      = cv2.resize(image2, (512, 512), interpolation=cv2.INTER_LINEAR)/255.0
    image2      = (image2-mean) / std


    model.eval()
    Image = torch.zeros(1, 2, 3, 512, 512)
    Image[:, 0, : ,: ,:] = torch.from_numpy(image1).permute(2,1,0).unsqueeze(0).float()
    Image[:, 1, : ,: ,:] = torch.from_numpy(image2).permute(2,1,0).unsqueeze(0).float()
    s_hm, p_hm, k_hm, b_hm = model.draw_heatmap(Image)
    # s, k, f = model.draw_heatmap(Image)

    # plt.subplot(153)   
    # img = s[0]
    # plt.imshow(img.permute(2,1,0).detach().numpy(), cmap='jet')
    # plt.axis('off')
    # plt.title('support heatmap', size=10)
    
    # plt.subplot(154)
    # img = k[0]
    # plt.imshow(img.permute(2,1,0).detach().numpy(), cmap='jet')
    # plt.axis('off')
    # plt.title('current heatmap', size=10)
    
    # plt.subplot(155) #
    # img = f[0]
    # plt.imshow(img.permute(2,1,0).detach().numpy(), cmap='jet')
    # plt.axis('off')
    # plt.title('fusion heatmap', size=10)
    

    plt.subplot(163)   
    img = s_hm[0]
    # s_min = np.min(s)
    # s_max = np.max(s)
    # s = (((s - s_min)) / (s_max - s_min + 0.0001)) * 255
    # # s = s.astype(np.uint8)
    # s = np.uint8(s)
    # s = cv2.applyColorMap(s, cv2.COLORMAP_JET)
    # s = s.permute(2, 1, 0)
    # plt.imshow(s)
    # plt.imshow((s.permute(2, 1, 0) * 255).detach().numpy().astype(np.uint8), cmap='gray') 
    plt.imshow(img.permute(2,1,0).detach().numpy(), cmap='jet')
    plt.axis('off')
    plt.title('support heatmap', size=10)
    
    plt.subplot(164)
    img = p_hm[0]
    # p_min = np.min(p)
    # p_max = np.max(p)
    # p = (((p - p_min)) / (p_max - p_min + 0.0001)) * 255
    # p = p.astype(np.uint8)
    # p = cv2.applyColorMap(p, cv2.COLORMAP_JET)
    # p = p.permute(2, 1, 0)
    # plt.imshow(p)
    # plt.imshow((p.permute(2, 1, 0) * 255).detach().numpy().astype(np.uint8), cmap='gray') 
    plt.imshow(img.permute(2,1,0).detach().numpy(), cmap='jet')
    plt.axis('off')
    plt.title('propagation heatmap', size=10)
    
    plt.subplot(165) #
    img = k_hm[0]
    # k_min = np.min(k)
    # k_max = np.max(k)
    # k = (((k - k_min)) / (k_max - k_min + 0.0001)) * 255
    # k = k.astype(np.uint8)
    # k = cv2.applyColorMap(k, cv2.COLORMAP_JET)
    # k = k.permute(2, 1, 0)
    # plt.imshow(k)
    # plt.imshow((k.permute(2, 1, 0) * 255).detach().numpy().astype(np.uint8), cmap='gray')
    plt.imshow(img.permute(2,1,0).detach().numpy(), cmap='jet')
    plt.axis('off')
    plt.title('current heatmap', size=10)
    
    plt.subplot(166)
    img = b_hm[0]
    # b_min = np.min(b)
    # b_max = np.max(b)
    # b = (((b - b_min)) / (b_max - b_min + 0.0001)) * 255
    # b = b.astype(np.uint8)
    # b = cv2.applyColorMap(b, cv2.COLORMAP_JET)
    # b = b.permute(2, 1, 0)
    # plt.imshow(b)
    # plt.imshow((b.permute(2, 1, 0) * 255).detach().numpy().astype(np.uint8), cmap='gray')  
    plt.imshow(img.permute(2,1,0).detach().numpy(), cmap='jet')
    plt.axis('off')
    plt.title('balanced heatmap', size=10)
    plt.savefig(save_path)
    plt.show()
    print('successful save heatmaps!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone'   , type=str   , default='pvt_v2_b2'              )
    parser.add_argument('--snapshot'   , type=str   , default='/mntnfs/med_data4/yuncheng/centernet/centernet/model/centernet_zs_vid_ph_alpha0_5_9/19.pth'                    )
    parser.add_argument('--epoch'      , type=int   , default=20                       )
    parser.add_argument('--time_clips' , type=int   , default=2                        )
    parser.add_argument('--alpha'      , type=int   , default=0.5                      )
    parser.add_argument('--lr'         , type=float , default=1e-4                     )
    parser.add_argument('--batch_size' , type=int   , default=16                       )
    parser.add_argument('--data_path'  , type=str   , default='../dataset/'            )
    parser.add_argument('--model_path' , type=str   , default='./model/'               )
    args = parser.parse_args()

    model       = PolypModel(args)
    # model        = PolypModelV2(args)
    # model        = PolypModelV3(args)
    img_path1    = '/mntnfs/med_data4/yuncheng/DATASET/ZSPolyp/test/29/0001.jpg'
    img_path2    = '/mntnfs/med_data4/yuncheng/DATASET/ZSPolyp/test/29/0002.jpg'
    save_path   = './figures/CAM_test_9.jpg'
    draw_CAM(model, img_path1, img_path2, save_path)