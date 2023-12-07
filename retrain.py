import os
import sys
import cv2
import time
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from tabulate import tabulate
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

from model import CenterNet, ECCPolypDet
from data import TrainData, ReTrainData
from utils import focal_loss, reg_l1_loss, ModelEma, focal_loss_weighted, reg_l1_loss_weighted
from utils import decode_bbox, postprocess

class Train:
    def __init__(self, args, exp_path):
        self.args         = args
        self.logger       = SummaryWriter(exp_path)
        ## data
        self.data         = TrainData(args)
        self.loader       = DataLoader(dataset=self.data, shuffle=True, batch_size=args.batch_size, num_workers=6, drop_last=True)
        ## model
        model_dict = {
            'CenterNet': CenterNet,
            'ECCPolypDet': ECCPolypDet,
        }
        if args.model_name in model_dict:
            self.model = model_dict[args.model_name](args).cuda()
        else:
            raise ValueError(f"Model name '{args.model_name}' not found in model_dict")
        ## optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), args.lr, weight_decay=5e-4)

        ## learning rate scheduler
        scheduler_dict = {
            'step': torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [20, 40, 60, 80], gamma=0.5, last_epoch=-1, verbose=False),
            'cos': torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epoch, eta_min=1e-5),
            'exp': torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5, last_epoch=-1, verbose=False)
        }
        self.scheduler = scheduler_dict.get(self.args.scheduler)
        ## model ema
        self.ema = ModelEma(self.model, decay=0.9998)

    def train(self):
        best_acc = 0
        global_step = 0
        valer = Validation(self.args.data_path, self.args.model_name)
        for epoch in range(1, args.epoch):
            self.model.train()
            if epoch<2:
                for param in self.model.backbone.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.backbone.parameters():
                    param.requires_grad = True
                
            for i, (images, hms, hms_i, whs, regs, reg_masks, masks, weight) in enumerate(self.loader):
                images, hms, hms_i, whs, regs, reg_masks, weight = images.cuda(), hms.cuda(), hms_i.cuda(), whs.cuda(), regs.cuda(), reg_masks.cuda(), weight.cuda()
                hm, wh, offset  = self.model(images)
                c_loss          = focal_loss_weighted(hm, hms, weight)
                wh_loss         = reg_l1_loss_weighted(wh, whs, reg_masks, weight)
                off_loss        = reg_l1_loss_weighted(offset, regs, reg_masks, weight)
                loss            = c_loss + wh_loss*0.1 + off_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema.update(self.model) 

                global_step += 1
                self.logger.add_scalar('lr'  , self.optimizer.param_groups[0]['lr'], global_step=global_step)
                self.logger.add_scalars('loss', {'loss':loss.item(), 'c_loss':c_loss.item(), 'wh_loss':wh_loss.item(), 'off_loss':off_loss.item()}, global_step=global_step)
                if i % 20 == 0:
                    print(f'{datetime.now()} Epoch={epoch:03d}/{self.args.epoch:03d}, Step={i:04d}/{len(self.loader):04d}, loss={loss.item():0.4f}, c_loss={c_loss.item():.4f}, wh_loss={wh_loss.item():.4f}, off_loss={off_loss.item():.4f}')
                    logging.info(f'{datetime.now()} Epoch={epoch:03d}/{self.args.epoch:03d}, Step={i:04d}/{len(self.loader):04d}, loss={loss.item():0.4f}, c_loss={c_loss.item():.4f}, wh_loss={wh_loss.item():.4f}, off_loss={off_loss.item():.4f}')

            precision, recall, f1, f2 = valer.val(self.model)
            self.logger.add_scalar('Metrics/Precision', precision, global_step=global_step)
            self.logger.add_scalar('Metrics/Recall'   , recall   , global_step=global_step)
            self.logger.add_scalar('Metrics/F1-score' , f1       , global_step=global_step)
            self.logger.add_scalar('Metrics/F2-score' , f2       , global_step=global_step)
            if f1 > best_acc:
                torch.save(self.model.state_dict(), os.path.join(exp_path, 'best.pth'))
                print(f"saved best model at epoch {epoch}, f1 score: {f1:.3f}")
                logging.info(f"{datetime.now()} saved best model at epoch {epoch}, f1 score: {f1:.3f}")
                best_acc = f1

            self.scheduler.step()


    def compute_contrast_loss(self, feature, masks, temperature=0.2):
        """
        Compute the contrastive loss for the given feature and masks.
        
        Args:
            feature (torch.Tensor): The feature tensor of shape [B, C, H, W].
            masks (torch.Tensor): The masks tensor of shape [B, H, W].
            temperature (float, optional): The temperature hyperparameter for the contrastive loss. Default is 0.2.
            
        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        masks       = masks.unsqueeze(1)  # mask : [B, 1, H, W]

        # Calculate positive and negative features
        pos         = (feature * masks).sum(dim=(2,3)) / (masks.sum(dim=(2,3)) + 1e-6)
        neg         = (feature * (1 - masks)).sum(dim=(2,3)) / ((1 - masks).sum(dim=(2,3)) + 1e-6)

        # Normalize positive and negative features
        pos         = F.normalize(pos, dim=1)
        neg         = F.normalize(neg, dim=1)

        # Compute positive-negative and positive-positive dot products
        pos_neg     = torch.mm(pos, neg.transpose(1,0)) #[B, B]
        pos_pos     = (pos * pos[torch.randperm(pos.size()[0])]).sum(dim=1) #[B, 1]

        # Compute logits and contrastive loss
        pos_logits  = torch.exp(pos_pos / temperature)
        neg_logits  = torch.exp(pos_neg / temperature)
        nce_loss    = (-torch.log(pos_logits / (pos_logits + neg_logits.sum(dim=1)))).mean(dim=0)

        return nce_loss

    '''
        contrast
    '''
    def train_contrast(self):
        best_acc    = 0
        global_step = 0
        valer       = Validation(self.args.data_path, self.args.model_name)
        
        # Loop through epochs
        for epoch in range(1, args.epoch):
            self.model.train()
            
            # Freeze or unfreeze backbone parameters based on epoch
            requires_grad = False if epoch < 2 else True
            for param in self.model.backbone.parameters():
                param.requires_grad = requires_grad


            # Loop through data loader
            for i, (images, hms, hms_i, whs, regs, reg_masks, masks, weight) in enumerate(self.loader):
                # Move tensors to GPU
                tensors = [images, hms, hms_i, whs, regs, reg_masks, masks, weight]
                images, hms, hms_i, whs, regs, reg_masks, masks, weight = (t.cuda() for t in tensors)
                
                # Forward pass and compute losses
                _, ct_feature, hm_i, hm, wh, offset = self.model(images)
                contrast_loss                       = self.compute_contrast_loss(ct_feature, masks)
                c_loss                              = focal_loss_weighted(hm, hms, weight)
                inter_loss                          = focal_loss_weighted(hm_i, hms_i, weight)
                wh_loss                             = reg_l1_loss_weighted(wh, whs, reg_masks, weight)
                off_loss                            = reg_l1_loss_weighted(offset, regs, reg_masks, weight)
                total_loss                          = c_loss + wh_loss * 0.1 + off_loss + contrast_loss * 0.3 + inter_loss * 0.3

                # Backward pass and optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.ema.update(self.model) 
                
                # Logging
                global_step += 1
                self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step=global_step)
                loss_dict = {'loss': total_loss.item(), 'c_loss': c_loss.item(), 'inter_loss': inter_loss.item(), 'contrast_loss': contrast_loss.item(), 'wh_loss': wh_loss.item(), 'off_loss': off_loss.item()}
                self.logger.add_scalars('loss', loss_dict, global_step=global_step)
                
                if i % 20 == 0:
                    print(f'{datetime.now()} Epoch={epoch:03d}/{self.args.epoch:03d}, Step={i:04d}/{len(self.loader):04d}, loss={total_loss.item():.4f}, c_loss={c_loss.item():.4f}, inter_loss={inter_loss.item():.4f}, contrast_loss={contrast_loss.item():.4f}, wh_loss={wh_loss.item():.4f}, off_loss={off_loss.item():.4f}')
                    logging.info(f'{datetime.now()} Epoch={epoch:03d}/{self.args.epoch:03d}, Step={i:04d}/{len(self.loader):04d}, loss={total_loss.item():.4f}, c_loss={c_loss.item():.4f}, inter_loss={inter_loss.item():.4f}, contrast_loss={contrast_loss.item():.4f}, wh_loss={wh_loss.item():.4f}, off_loss={off_loss.item():.4f}')
            
            # Validation
            precision, recall, f1, f2 = valer.val(self.model)
            self.logger.add_scalar('Metrics/Precision', precision, global_step=global_step)
            self.logger.add_scalar('Metrics/Recall', recall, global_step=global_step)
            self.logger.add_scalar('Metrics/F1-score', f1, global_step=global_step)
            self.logger.add_scalar('Metrics/F2-score', f2, global_step=global_step)
            
            # Save best model
            if f1 > best_acc:
                torch.save(self.model.state_dict(), os.path.join(exp_path, 'best.pth'))
                print(f"saved best model at epoch {epoch}, f1 score: {f1}")
                logging.info(f"saved best model at epoch {epoch}, f1 score: {f1}")
                best_acc = f1
            
            # save every epoch
            torch.save(self.model.state_dict(), os.path.join(exp_path, 'epoch_%d.pth'%epoch))

            self.scheduler.step()


def num_iou(bboxs, gt_bboxs):
    """
    Calculate the number of true positives and false positives based on Intersection over Union (IoU) threshold.
    
    Args:
        bboxs (list): List of predicted bounding boxes.
        gt_bboxs (list): List of ground truth bounding boxes.
        
    Returns:
        tuple: A tuple containing the number of true positives, false positives, total predicted boxes, and total ground truth boxes.
    """
    num_tp = 0
    for box in bboxs:
        flag = False
        for gt_box in gt_bboxs:
            xmin, ymin, xmax, ymax  = box
            x1, y1, x2, y2          = gt_box
            width, height           = max(min(xmax, x2) - max(xmin, x1), 0), max(min(ymax, y2) - max(ymin, y1), 0)
            union                   = (xmax - xmin) * (ymax - ymin) + (x2 - x1) * (y2 - y1)
            inter                   = width * height
            iou                     = inter / (union - inter + 1e-6)
            if 1 > iou > 0.5 and width > 0 and height > 0:       
                flag = True
                break
        if flag:
            num_tp += 1
    return num_tp, len(bboxs) - num_tp, len(bboxs), len(gt_bboxs)

class Validation:
    """
    A class to perform validation on a given model using a dataset.

    Attributes:
        data_path (str): The path to the dataset.
        model_name (str): The name of the model to be validated.
        test_mode (bool): Whether to run in test mode or not.
    """

    def __init__(self, data_path, model_name, test_mode=False):
        self.data_path  = data_path
        self.model_name = model_name
        self.mean       = np.array([0.485, 0.456, 0.406])
        self.std        = np.array([0.229, 0.224, 0.225])
        self.confidence = 0.3
        self.nms_iou    = 0.3
        self.names      = []
        self.samples    = []
        self._load_samples()
        self._load_images()

    def _load_samples(self):
        with open(self.data_path + '/test_hard_box.txt') as lines:
            for line in lines:
                name, boxs = line.strip().split(';')
                boxs = boxs.split(' ')
                bbox = []
                for i in range(len(boxs) // 4):
                    xmin, ymin, xmax, ymax = boxs[4 * i:4 * (i + 1)]
                    bbox.append([max(int(xmin), 0), max(int(ymin), 0), int(xmax), int(ymax)])
                self.samples.append([name, bbox])
        print('test samples:', len(self.samples))
        logging.info('test samples: ' + str(len(self.samples)))

    def _load_images(self):
        print("Loading testing dataset:")
        logging.info("Loading testing dataset:")
        self.Image  = []
        self.Bbox   = []
        self.Height = []
        self.Width  = []
        for idx, (name, bbox) in enumerate(self.samples):
            name    = self.data_path + '/TestHardDataset/Frame/' + name
            image   = cv2.imread(name)
            image   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            H, W, C = image.shape
            self.Image.append(image)
            self.Bbox.append(bbox)
            self.Height.append(H)
            self.Width.append(W)
            self._print_progress(idx)
        print('100%|' + '*' * 50 + ' | ' + str(idx) + '/' + str(len(self.samples)))
        logging.info('100%|' + '*' * 50 + ' | ' + str(idx) + '/' + str(len(self.samples)))
    def _print_progress(self, idx):
        """
        Prints the progress of loading images as a percentage.

        Args:
            idx (int): The current index of the image being loaded.
        """
        if idx == int(len(self.samples) * 0.3):
            print('30%|' + '*' * 15 + ' | ' + str(idx) + '/' + str(len(self.samples)))
            logging.info('30%|' + '*' * 15 + ' | ' + str(idx) + '/' + str(len(self.samples)))
        elif idx == int(len(self.samples) * 0.6):
            print('60%|' + '*' * 30 + ' | ' + str(idx) + '/' + str(len(self.samples)))
            logging.info('60%|' + '*' * 30 + ' | ' + str(idx) + '/' + str(len(self.samples)))
    def val(self, model):
        """
        Validates the model using the dataset.

        Args:
            model: The model to be validated.

        Returns:
            precision (float): The precision of the model.
            recall (float): The recall of the model.
            f1 (float): The F1 score of the model.
            f2 (float): The F2 score of the model.
        """
        self.model = model
        self.model.eval()
        with torch.no_grad():
            start = time.time()
            num_tps, num_fps, num_dets, num_gts = 0, 0, 0, 0
            for idx in range(len(self.samples)):
                image, gt_bboxs, H, W = self._prepare_image(idx)
                if self.model_name == 'CenterNet':
                    heatmap, whpred, offset = self.model(image)
                elif self.model_name == 'ECCPolypDet':
                    _, _, heat, heatmap, whpred, offset = self.model(image)

                outputs = decode_bbox(heatmap, whpred, offset, self.confidence)
                results = postprocess(outputs, (H, W), self.nms_iou)
                if results[0] is None:
                    num_gts += len(gt_bboxs)
                    continue
                confidences = results[0][:, 4]
                pred_box    = results[0][:, :4]
                bboxs = self._process_results(pred_box, confidences)
                num_tp, num_fp, num_det, num_gt = num_iou(bboxs, gt_bboxs)
                num_tps, num_fps, num_dets, num_gts = num_tps + num_tp, num_fps + num_fp, num_dets + num_det, num_gts + num_gt
        
        end = time.time()
        seconds = end - start
        fps  = len(self.samples) / seconds
        precision, recall, f1, f2 = self._calculate_metrics(num_tps, num_dets, num_gts)
        tables  = [[num_tps, num_fps, num_dets, num_gts, precision, recall, f1, f2, fps]]
        headers = ['num_tps', 'num_fps', 'num_dets', 'num_gts', 'precision' ,'recall', 'f1', 'f2', 'fps']
        print(tabulate(tables, headers, tablefmt="grid", numalign="center"))
        logging.info('\n'+tabulate(tables, headers, tablefmt="grid", numalign="center"))
        return precision, recall, f1, f2

    def _prepare_image(self, idx):
        """
        Prepares the image for validation by resizing and normalizing it.

        Args:
            idx (int): The index of the image in the dataset.

        Returns:
            image (tensor): The prepared image.
            gt_bboxs (list): The ground truth bounding boxes.
            H (int): The height of the image.
            W (int): The width of the image.
        """
        image       = self.Image[idx]
        gt_bboxs    = self.Bbox[idx]
        H           = self.Height[idx]
        W           = self.Width[idx]
        image       = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR) / 255.0
        image       = (image - self.mean) / self.std
        image       = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda().float()
        return image, gt_bboxs, H, W

    def _process_results(self, pred_box, confidences):
        """
        Processes the results of the model by converting them to bounding boxes.

        Args:
            results (list): The results of the model.

        Returns:
            bboxs (list): The list of bounding boxes.
        """
        bboxs = []
        for box, confidence in zip(pred_box, confidences):
            if confidence > 0.1:
                ymin, xmin, ymax, xmax = box
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                bboxs.append([xmin, ymin, xmax, ymax])
        return bboxs

    def _calculate_metrics(self, num_tps, num_dets, num_gts):
        """
        Calculates the precision, recall, F1, and F2 scores.

        Args:
            num_tps (int): The number of true positives.
            num_dets (int): The number of detections.
            num_gts (int): The number of ground truths.

        Returns:
            precision (float): The precision of the model.
            recall (float): The recall of the model.
            f1 (float): The F1 score of the model.
            f2 (float): The F2 score of the model.
        """
        precision   = num_tps / (num_dets + 1e-6)
        recall      = num_tps / num_gts
        f1          = 2 * num_tps / (num_dets + num_gts)
        f2          = (5 * num_tps) / (4 * num_gts + num_dets)
        return precision, recall, f1, f2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_path', type=str, default='../dataset/')
    parser.add_argument('--save_path', type=str, default='./result/')
    parser.add_argument('--model_name', type=str, default='PolypModel')
    args = parser.parse_args()

    ## experiment path ##
    save_path          = os.path.join(args.save_path, args.model_name)
    current_timestamp  = datetime.now().timestamp()
    current_datetime   = datetime.fromtimestamp(current_timestamp+29220)  # different time zone
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    exp_path           = os.path.join(save_path, 'log_' + formatted_datetime)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(exp_path, exist_ok=True)

    trainer = Train(args, exp_path)


    logging.basicConfig(filename=exp_path+'/log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Config")
    table = [[args.model_name, args.backbone, args.epoch, args.batch_size, args.lr, args.scheduler, torch.cuda.get_device_name(0)]]
    headers = ["Model", "Backbone", "Epoch", "Batch", "LR", "Scheduler", "GPU"]
    print(tabulate(table, headers, tablefmt="grid", numalign="center"))
    logging.info('\n'+tabulate(table, headers, tablefmt="grid", numalign="center"))

    model_train_methods = {
        'CenterNet': trainer.train,
        'ECCPolypDet': trainer.train_contrast,
    }

    start_time = time.time()
    if args.model_name in model_train_methods:
        model_train_methods[args.model_name]()
    else:
        print(f"Invalid model name: {args.model_name}")

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print(f'Training completed.\nTotal training time: {format(total_time_str)}')