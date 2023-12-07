import os
import torch
import math
import random
import numpy as np
from torch import nn
from torchvision.ops import nms
import torch.nn.functional as F
from copy import deepcopy

def focal_loss(pred, target):
    pred = pred.squeeze()
    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    pos_inds = (target==1).float()
    neg_inds = (target<1).float()
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)
    
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = torch.log(pred)*torch.pow(1-pred, 2)*pos_inds
    neg_loss = torch.log(1-pred)*torch.pow(pred, 2)*neg_weights*neg_inds
    
    #-------------------------------------------------------------------------#
    #   进行损失的归一化
    #-------------------------------------------------------------------------#
    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    loss     = -(pos_loss + neg_loss) / num_pos
    return loss

def focal_loss_weighted(pred, target, weight):
    pred = pred.squeeze()
    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    pos_inds = (target==1).float()
    neg_inds = (target<1).float()
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)
    
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    weight = weight[:, None, None, None]
    pos_loss = torch.log(pred)*torch.pow(1-pred, 2)*pos_inds * weight
    neg_loss = torch.log(1-pred)*torch.pow(pred, 2)*neg_weights*neg_inds * weight
    
    #-------------------------------------------------------------------------#
    #   进行损失的归一化
    #-------------------------------------------------------------------------#
    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    loss     = -(pos_loss + neg_loss) / num_pos
    return loss

def reg_l1_loss(pred, target, mask):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    pred        = pred.permute(0,2,3,1)
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,2)
    loss        = F.l1_loss(pred*expand_mask, target*expand_mask, reduction='sum')
    loss        = loss / (mask.sum() + 1e-4)
    return loss

def reg_l1_loss_weighted(pred, target, mask, weight):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    pred            = pred.permute(0,2,3,1)
    weight          = weight[:, None, None, None]
    expand_mask     = torch.unsqueeze(mask,-1).repeat(1,1,1,2)
    loss            = F.l1_loss(pred*expand_mask, target*expand_mask, reduction='none')
    loss            = loss * weight
    loss            = loss.sum()
    loss            = loss / (mask.sum() + 1e-4)
    return loss

def pool_nms(heat, kernel = 3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def decode_bbox(pred_hms, pred_whs, pred_offsets, confidence):
    #-------------------------------------------------------------------------#
    #   当利用512x512x3图片进行coco数据集预测的时候
    #   h = w = 128 num_classes = 80
    #   Hot map热力图 -> b, 80, 128, 128, 
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    #-------------------------------------------------------------------------#
    pred_hms = pool_nms(pred_hms)
    
    b, c, output_h, output_w = pred_hms.shape
    detects = []
    #-------------------------------------------------------------------------#
    #   只传入一张图片，循环只进行一次
    #-------------------------------------------------------------------------#
    for batch in range(b):
        #-------------------------------------------------------------------------#
        #   heat_map        128*128, num_classes    热力图
        #   pred_wh         128*128, 2              特征点的预测宽高
        #                                           在预测过程的前处理以及后处理视频中讲的有点小问题，不是调整参数，就是宽高
        #   pred_offset     128*128, 2              特征点的xy轴偏移情况
        #-------------------------------------------------------------------------#
        heat_map    = pred_hms[batch].permute(1, 2, 0).view([-1, c])
        pred_wh     = pred_whs[batch].permute(1, 2, 0).view([-1, 2])
        pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])

        yv, xv      = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w), indexing='ij')
        #-------------------------------------------------------------------------#
        #   xv              128*128,    特征点的x轴坐标
        #   yv              128*128,    特征点的y轴坐标
        #-------------------------------------------------------------------------#
        xv, yv      = xv.cuda().flatten().float(), yv.cuda().flatten().float()

        #-------------------------------------------------------------------------#
        #   class_conf      128*128,    特征点的种类置信度
        #   class_pred      128*128,    特征点的种类
        #-------------------------------------------------------------------------#
        class_conf, class_pred  = torch.max(heat_map, dim = -1)
        mask                    = class_conf > confidence

        #-----------------------------------------#
        #   取出得分筛选后对应的结果
        #-----------------------------------------#
        pred_wh_mask        = pred_wh[mask]
        pred_offset_mask    = pred_offset[mask]
        if len(pred_wh_mask) == 0:
            detects.append([])
            continue     

        #----------------------------------------#
        #   计算调整后预测框的中心
        #----------------------------------------#
        xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
        #----------------------------------------#
        #   计算预测框的宽高
        #----------------------------------------#
        half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
        #----------------------------------------#
        #   获得预测框的左上角和右下角
        #----------------------------------------#
        bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h
        bboxes[:, 0][bboxes[:, 0] < 0] = 0
        bboxes[:, 1][bboxes[:, 1] < 0] = 0 
        bboxes[:, 2][bboxes[:, 2] > output_w - 1] = output_w - 1
        bboxes[:, 3][bboxes[:, 3] > output_h - 1] = output_h - 1
        detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)
        detects.append(detect)
    return detects

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
                 
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    iou = inter_area / torch.clamp(b1_area + b2_area - inter_area, min = 1e-6)

    return iou

def centernet_correct_boxes(box_xy, box_wh, image_shape):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    image_shape = np.array(image_shape)
    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def postprocess(prediction, image_shape, nms_thres=0.4):
    output = [None for _ in range(len(prediction))]
    
    #----------------------------------------------------------#
    #   预测只用一张图片，只会进行一次
    #----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        detections      = prediction[i]
        if len(detections) == 0:
            continue
        #------------------------------------------#
        #   获得预测结果中包含的所有种类
        #------------------------------------------#
        unique_labels   = detections[:, -1].cpu().unique()

        if detections.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            #------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            #------------------------------------------#
            detections_class = detections[detections[:, -1] == c]
            #------------------------------------------#
            #   使用官方自带的非极大抑制会速度更快一些！
            #------------------------------------------#
            # keep = nms(
            #     detections_class[:, :4],
            #     detections_class[:, 4],
            #     nms_thres
            # )
            # max_detections = detections_class[keep]

            #------------------------------------------#
            #   按照存在物体的置信度排序
            #------------------------------------------#
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            #------------------------------------------#
            #   进行非极大抑制
            #------------------------------------------#
            max_detections = []
            while detections_class.size(0):
                #---------------------------------------------------#
                #   取出这一类置信度最高的，一步一步往下判断。
                #   判断重合程度是否大于nms_thres，如果是则去除掉
                #---------------------------------------------------#
                max_detections.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]
            #------------------------------------------#
            #   堆叠
            #------------------------------------------#
            max_detections = torch.cat(max_detections).data
            
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i]           = output[i].cpu().numpy()
            box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4]    = centernet_correct_boxes(box_xy, box_wh, image_shape)
    return output

class ModelEma(nn.Module):
    """ Model Exponential Moving Average V2
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)




if __name__ == '__main__':
    pred    = torch.ones(1,2,128,128)
    target  = torch.ones(1,128,128,2)
    mask    = torch.ones(1,128,128)
    loss    = reg_l1_loss(pred, target, mask)
