import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tqdm import tqdm
from pathlib import Path

def anno2coco_sch(path):
    ''' COCO Format
    {
        "images": [image],
        "annotations": [annotation],
        "categories": [category]
    }

    image = {
        "id": int,
        "width": int,
        "height": int,
        "file_name": str,
    }

    annotation = {
        "id": int,
        "image_id": int,
        "category_id": int,
        "segmentation": RLE or [polygon],
        "area": float,
        "bbox": [x,y,width,height],
        "iscrowd": 0 or 1,
    }

    categories = [{
        "id": int,
        "name": str,
        "supercategory": str,
    }]
    '''
    images, annotations, categories = [], [], []
    image_id, object_id             = 0, 0
    for folder in sorted(os.listdir(path), key=lambda x:int(x)):
        print(folder)
        for name in sorted(os.listdir(path+'/'+folder + '/clear')):
            if '.json' in name:
                ## image
                image = cv2.imread(path+'/'+folder+'/'+ 'clear/' + name.replace('.json', '.jpg'))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                H,W,C = image.shape
                images.append({ 
                    "id"        : image_id, 
                    "width"     : W, 
                    "height"    : H, 
                    "file_name" : folder+'/'+name.replace('.json', '.jpg')
                })

                ## annotation
                with open(path+'/'+folder+'/'+ 'clear/' + name, 'r') as f:
                    anno = json.load(f)
                    for shape in anno['shapes']:
                        points     = np.array(shape['points'])
                        xmin, ymin = np.min(points, axis=0)
                        xmax, ymax = np.max(points, axis=0)
                        annotations.append({
                            "id"            : object_id,
                            "image_id"      : image_id,
                            "category_id"   : 0,
                            "segmentation"  : [points.flatten().tolist()],
                            "area"          : (xmax-xmin)*(ymax-ymin),
                            "bbox"          : [xmin, ymin, xmax-xmin, ymax-ymin],
                            "iscrowd"       : 0,
                        })
                        object_id += 1
                image_id += 1

    ## category
    categories.append({
        "id"           : 0,
        "name"         : 'polyp',
        "supercategory": 'polyp',
    })
    coco_format_json = {
        "images"      : images,
        "annotations" : annotations,
        "categories"  : categories
    }
    with open('/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp/'+'clear_test.json', 'w+') as f:
        json.dump(coco_format_json, f)

def anno2coco_sysu(path):
    ''' COCO Format
    {
        "images": [image],
        "annotations": [annotation],
        "categories": [category]
    }

    image = {
        "id": int,
        "width": int,
        "height": int,
        "file_name": str,
    }

    annotation = {
        "id": int,
        "image_id": int,
        "category_id": int,
        "segmentation": RLE or [polygon],
        "area": float,
        "bbox": [x,y,width,height],
        "iscrowd": 0 or 1,
    }

    categories = [{
        "id": int,
        "name": str,
        "supercategory": str,
    }]
    '''
    images, annotations, categories = [], [], []
    image_id, object_id             = 0, 0
    for folder in sorted(os.listdir(path)):
        if folder == '.DS_Store':
            continue
        print(folder)
        for name in sorted(os.listdir(path+'/'+folder)):
            if '.json' in name:
                ## image
                image = cv2.imread(path+'/'+folder+'/' + name.replace('.json', '.jpg'))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                H,W,C = image.shape
                images.append({ 
                    "id"        : image_id, 
                    "width"     : W, 
                    "height"    : H, 
                    "file_name" : folder+'/'+name.replace('.json', '.jpg')
                })

                ## annotation
                with open(path+'/'+folder+'/'+ name, 'r') as f:
                    anno = json.load(f)
                    for shape in anno['shapes']:
                        points     = np.array(shape['points'])
                        xmin, ymin = np.min(points, axis=0)
                        xmax, ymax = np.max(points, axis=0)
                        annotations.append({
                            "id"            : object_id,
                            "image_id"      : image_id,
                            "category_id"   : 0,
                            "segmentation"  : [points.flatten().tolist()],
                            "area"          : (xmax-xmin)*(ymax-ymin),
                            "bbox"          : [xmin, ymin, xmax-xmin, ymax-ymin],
                            "iscrowd"       : 0,
                        })
                        object_id += 1
                image_id += 1

    ## category
    categories.append({
        "id"           : 0,
        "name"         : 'polyp',
        "supercategory": 'polyp',
    })
    coco_format_json = {
        "images"      : images,
        "annotations" : annotations,
        "categories"  : categories
    }
    with open('/mntnfs/med_data4/yuncheng/DATASET/SCH_ZSPolyp/'+'train.json', 'w+') as f:
        json.dump(coco_format_json, f)


def anno2coco_sunseg():
    ''' COCO Format
    {
        "images": [image],
        "annotations": [annotation],
        "categories": [category]
    }

    image = {
        "id": int,
        "width": int,
        "height": int,
        "file_name": str,
    }

    annotation = {
        "id": int,
        "image_id": int,
        "category_id": int,
        "segmentation": RLE or [polygon],
        "area": float,
        "bbox": [x,y,width,height],
        "iscrowd": 0 or 1,
    }

    categories = [{
        "id": int,
        "name": str,
        "supercategory": str,
    }]
    '''
    images, annotations, categories = [], [], []
    with open('/mntnfs/med_data4/yuncheng/DATASET/SUN-SEG/TrainDataset/Detection/bbox_annotation.json', 'r') as f:
        anno = json.load(f)
        images_list = anno['images']
        annotations_list = anno['annotation']
        image_id = 0
        for image in images_list:
            id = image["id"]
            W, H = image["width"], image["height"]
            file_name = image["file_name"]
            file_name = id.split('-')[0] + '/' + file_name + '.jpg'
            images.append({
                    "id"        : image_id, 
                    "width"     : W, 
                    "height"    : H, 
                    "file_name" : file_name
                        })
            image_id += 1

        image_id, object_id             = 0, 0
        for annotation in annotations_list:
            id = annotation["id"]
            ymin, xmin, width, height = annotation["bbox"]
            xmax = xmin + width
            ymax = ymin + height
            annotations.append({
                            "id"            : object_id,
                            "image_id"      : image_id,
                            "category_id"   : 0,
                            "segmentation"  : [],
                            "area"          : (xmax-xmin)*(ymax-ymin),
                            "bbox"          : [xmin, ymin, width, height],
                            "iscrowd"       : 0,
                        })
            image_id += 1
            object_id += 1
    ## category
    categories.append({
        "id"           : 0,
        "name"         : 'polyp',
        "supercategory": 'polyp',
    })
    coco_format_json = {
        "images"      : images,
        "annotations" : annotations,
        "categories"  : categories
    }
    with open('/mntnfs/med_data4/yuncheng/DATASET/SUN-SEG/'+'train.json', 'w+') as f:
        json.dump(coco_format_json, f)

def anno2coco_smallpolyp(path):
    ''' COCO Format
    {
        "images": [image],
        "annotations": [annotation],
        "categories": [category]
    }

    image = {
        "id": int,
        "width": int,
        "height": int,
        "file_name": str,
    }

    annotation = {
        "id": int,
        "image_id": int,
        "category_id": int,
        "segmentation": RLE or [polygon],
        "area": float,
        "bbox": [x,y,width,height],
        "iscrowd": 0 or 1,
    }

    categories = [{
        "id": int,
        "name": str,
        "supercategory": str,
    }]
    '''
    images, annotations, categories = [], [], []
    image_id, object_id             = 0, 0
    for name in tqdm(sorted(os.listdir(path+'/images'))):
        # image
        image = cv2.imread(path+'/images/'+name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H,W,C = image.shape
        images.append({ 
            "id"        : image_id, 
            "width"     : W, 
            "height"    : H, 
            "file_name" : name
        })

        # annotation
        mask     = cv2.imread(path+'/mask/'+name.replace('.jpg', '.png'), cv2.IMREAD_GRAYSCALE)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            xmin, ymin, xmax, ymax = x, y, x+w, y+h
            annotations.append({
                "id"            : object_id,
                "image_id"      : image_id,
                "category_id"   : 0,
                "segmentation"  : [],
                "area"          : (xmax-xmin)*(ymax-ymin),
                "bbox"          : [xmin, ymin, xmax-xmin, ymax-ymin],
                "iscrowd"       : 0,
            })
            object_id += 1
        image_id += 1

    ## category
    categories.append({
        "id"           : 0,
        "name"         : 'polyp',
        "supercategory": 'polyp',
    })
    coco_format_json = {
        "images"      : images,
        "annotations" : annotations,
        "categories"  : categories
    }
    with open('/mntnfs/med_data5/lizhuo/polyp_data/overall_images/'+'test.json', 'w+') as f:
        json.dump(coco_format_json, f)






def save_gt_sch(path):
    dets = []
    for folder in sorted(os.listdir(path), key=lambda x:int(x)):
        print(folder) 
        for name in sorted(os.listdir(path + '/' + folder + '/clear')):
            if '.json' in name: 
                with open(path+'/'+folder+'/clear/'+ name, 'r') as f:
                    anno = json.load(f)
                    boxes = []
                    boxes_of_current_image = {}
                    store = False
                    for shape in anno['shapes']:
                        if shape is not None:
                            store = True
                            points     = np.array(shape['points'])
                            xmin, ymin = np.min(points, axis=0)
                            xmax, ymax = np.max(points, axis=0)
                            boxes.append({
                                    "xtl": xmin,
                                    "xbr": xmax,
                                    "ytl": ymin,
                                    "ybr": ymax,
                                    "label": "mask",
                                    "score":1
                                    })
                            boxes_of_current_image = {"name":folder + '/' + name.replace('.json', '.jpg'), 'boxes': boxes}
                    if store == True:
                        dets.append(boxes_of_current_image)
    with open('/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp/'+'gt.json', 'w') as ff:
        json.dump(dets, ff, indent = 2)

def save_gt_sysu(path):
    dets = []
    for folder in sorted(os.listdir(path), key=lambda x:int(x)):
        print(folder) 
        for name in sorted(os.listdir(path + '/' + folder)):
            if '.json' in name: 
                with open(path + '/' + folder + '/' + name, 'r') as f:
                    anno = json.load(f)
                    boxes = []
                    boxes_of_current_image = {}
                    store = False
                    for shape in anno['shapes']:
                        if shape is not None:
                            store = True
                            points     = np.array(shape['points'])
                            xmin, ymin = np.min(points, axis=0)
                            xmax, ymax = np.max(points, axis=0)
                            boxes.append({
                                    "xtl": xmin,
                                    "xbr": xmax,
                                    "ytl": ymin,
                                    "ybr": ymax,
                                    "label": "mask",
                                    "score":1
                                    })
                            boxes_of_current_image = {"name":folder + '/' + name.replace('.json', '.jpg'), 'boxes': boxes}
                    if store == True:
                        dets.append(boxes_of_current_image)
    with open('/mntnfs/med_data4/yuncheng/DATASET/ZSPolyp/'+'gt2.json', 'w') as ff:
        json.dump(dets, ff, indent = 2)
                    
def save_gt_sun(path):
    dets = []
    
    with open(path + '/TestHardDataset/Detection/bbox_annotation.json', 'r') as f:
        anno = json.load(f)
        boxes = []
        id = 0
        for annotation in tqdm(anno['annotation']):
            name  = anno['images'][id]["id"].split('-')[0] + '/' + anno['images'][id]["file_name"] + '.jpg'
            id += 1
            ymin, xmin, width, height = annotation["bbox"]
            xmax = xmin + width
            ymax = ymin + height
            boxes = [
                    {
                    "xtl": xmin,
                    "xbr": xmax,
                    "ytl": ymin,
                    "ybr": ymax,
                    "label": "mask",
                    "score":1   
                    }
                    ]
            dets.append({"name":name, 'boxes': boxes})
    with open('/mntnfs/med_data4/yuncheng/DATASET/SUN-SEG/'+'gt_hard.json', 'w') as ff:
        json.dump(dets, ff, indent = 2)

def show_anno(path):
    coco   = COCO(path+'.json')
    imgIds = coco.getImgIds()
    imgs   = coco.loadImgs(imgIds)
    np.random.shuffle(imgs)
    for img in imgs:
        image = cv2.imread(path+'/'+img['file_name'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annIds = coco.getAnnIds(imgIds=img['id'])
        anns   = coco.loadAnns(annIds)
        
        plt.subplot(221)
        plt.imshow(image)
        plt.subplot(222)
        print(anns)
        mask   = coco.annToMask(anns)
        plt.imshow(mask)
        plt.subplot(223)
        plt.imshow(image)
        coco.showAnns(anns)
        plt.savefig('tt.png')
        input()
        plt.cla()


def name_list(path):
    with open(path+'.txt', 'w') as f:
        for folder in os.listdir(path):
            for name in os.listdir(path+'/'+folder):
                if '.json' in name:
                    f.write(folder+'/'+name.replace('.json', '\n'))

def save_color(path):
    color_list = []
    for folder in sorted(os.listdir(path), key=lambda x: int(x)):
        print(folder)
        for name in os.listdir(path+'/'+folder):
            if '.json' in name:
                image     = cv2.imread(path+'/'+folder+'/'+name.replace('.json', '.jpg'))
                image     = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                mean, std = image.mean(axis=(0,1), keepdims=True), image.std(axis=(0,1), keepdims=True)
                color_list.append((mean, std))
    np.save('color.npy', color_list)


# def save_box(path):
#     with open('/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp/clear_test_box.txt', 'w') as fbox:
#         folders = os.listdir(path)
#         # folders.remove('.DS_Store')
#         for folder in sorted(folders, key=lambda x:int(x)):
#             print(folder)
#             for name in sorted(os.listdir(path+'/'+folder)):
#                 if '.json' in name:
#                     image = cv2.imread(path+'/'+folder+'/'+name.replace('.json', '.jpg'))
#                     H,W,C = image.shape
#                     line  = folder+'/'+name.replace('.json', '.jpg')+';'
#                     print(path+'/'+folder+'/'+name)
#                     with open(path+'/'+folder+'/'+name, 'r') as f:
#                         anno = json.load(f)
#                         for shape in anno['shapes']:
#                             points     = np.array(shape['points'])
#                             xmin, ymin = np.min(points, axis=0)
#                             xmax, ymax = np.max(points, axis=0)
#                             xmin, ymin, xmax, ymax = max(int(xmin), 0), max(int(ymin), 0), min(int(xmax), W-1), min(int(ymax), H-1)
#                             line       = line+str(int(xmin))+' '+str(int(ymin))+' '+str(int(xmax))+' '+str(int(ymax))+' '
#                     if line[-1]!=';':
#                         fbox.write(line[:-1]+'\n')

def save_box(path):
    with open('/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp/clear_test_box.txt', 'w') as fbox:
        folders = os.listdir(path)
        # folders.remove('.DS_Store')
        for folder in sorted(folders, key=lambda x:int(x)):
            print(folder)
            for label in os.listdir(path+'/'+folder):
                if label == 'clear':
                    for name in sorted(os.listdir(path+'/'+folder+'/'+label)):
                        if '.json' in name:
                            image = cv2.imread(path+'/'+folder+'/'+label+'/'+name.replace('.json', '.jpg'))
                            H,W,C = image.shape
                            line  = folder+'/'+name.replace('.json', '.jpg')+';'
                            print(path+'/'+folder+'/'+label+'/'+name)
                            with open(path+'/'+folder+'/'+label+'/'+name, 'r') as f:
                                anno = json.load(f)
                                for shape in anno['shapes']:
                                    points     = np.array(shape['points'])
                                    xmin, ymin = np.min(points, axis=0)
                                    xmax, ymax = np.max(points, axis=0)
                                    xmin, ymin, xmax, ymax = max(int(xmin), 0), max(int(ymin), 0), min(int(xmax), W-1), min(int(ymax), H-1)
                                    line       = line+str(int(xmin))+' '+str(int(ymin))+' '+str(int(xmax))+' '+str(int(ymax))+' '
                            if line[-1]!=';':
                                fbox.write(line[:-1]+'\n')

# save_lst for sch polyp dataset
def save_lst_sch(path):
    with open('/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp/test.txt', 'w') as f:
        folders = os.listdir(path)
        for folder in sorted(folders, key=lambda x:int(x)):
            print(folder)
            for label in os.listdir(path + '/' + folder):
                if label == 'clear':
                    for name in sorted(os.listdir(path+'/'+folder+'/'+label)):
                        if '.json' in name:
                            with open(path+'/'+folder+'/'+ label + '/' + name, 'r') as ff:
                                anno = json.load(ff)
                                if anno['shapes'] == None:
                                    continue
                                f.write('/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp/test' + '/' + folder + '/' + name.replace('.json', '.jpg') + '\n')

# save_lst for sysu polyp dataset
def save_lst_sysu(path):
    with open(path + '/train.txt', 'w') as f:
        folders = os.listdir(path + '/train')
        for folder in sorted(folders, key=lambda x:int(x)):
            print(folder)
            for name in sorted(os.listdir(path + '/train/' + folder)):
                if '.json' in name:
                    with open(path + '/train/' + folder + '/' + name, 'r') as ff:
                        anno = json.load(ff)
                        if anno['shapes'] == None:
                            continue
                        f.write(path + '/train/' + folder + '/' + name.replace('.json', '.jpg') + '\n')

# save_lst for sun-seg polyp dataset

def save_lst_sun(path):
    with open(path + '/test_easy.txt', 'w') as f:
        folders = os.listdir(path + '/TestEasyDataset/Frame')
        print(folders)
        for i in range(len(folders)):
            if len(folders[i][4:].split('_')[0]) == 1:
                folders[i] = 'case000' +folders[i][4:]
            elif len(folders[i][4:].split('_')[0]) == 2:
                folders[i] = 'case00' + folders[i][4:]
            else:
                folders[i] = 'case0' + folders[i][4:]
        # for folder in tqdm(sorted(folders, key=lambda x:int(x[4:].split('_')[0]+'.'+x[4:].split('_')[1]))):
        for folder in sorted(folders, key=lambda x:x[4:]):
            folder = 'case' + folder[4:].lstrip('0')
            for name in sorted(os.listdir(path + '/TestEasyDataset/Frame/' + folder)):
                f.write(path + '/TestEasyDataset/Frame/' + folder + '/' + name + '\n')

def show():
    img = cv2.imread('/mntnfs/med_data4/yuncheng/DATASET/SUN-SEG/TrainDataset/Frame/case1_1/case_M_20181001100941_0U62372100109341_1_005_001-1_a2_ayy_image0001.jpg')
    H,W,C = img.shape
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread('/mntnfs/med_data4/yuncheng/DATASET/SUN-SEG/TrainDataset/GT/case1_1/case_M_20181001100941_0U62372100109341_1_005_001-1_a2_ayy_image0001.png')
    xmin, ymin, xmax, ymax =  [72,262,140,343]
    # xmin, ymin, xmax, ymax =  [
    #             691,
    #             925,
    #             72,
    #             111
    #         ]

    # origin = cv2.rectangle(img, (50,250), (180,350), color=(0,255,0), thickness=5)
    origin = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color=(0,255,0), thickness=5)
    # origin = cv2.rectangle(img, (ymin,xmin), (ymin+xmax,xmin+ymax), color=(0,255,0), thickness=5)
    plt.figure()
    plt.imshow(np.uint8(origin * 0.5 + mask * 0.5))
    plt.savefig('sunseg5.png')


'''
SUN SEG dataset
'''
def save_box():
    with open('/mntnfs/med_data4/yuncheng/DATASET/SUN-SEG/test_easy_box.txt', 'w') as fbox:
        with open('/mntnfs/med_data4/yuncheng/DATASET/SUN-SEG/TestEasyDataset/Detection/bbox_annotation.json', 'r') as f:
            files = json.load(f)
            for anno in tqdm(files['annotation']):
                idx = anno['id']
                for img in files['images']:
                    if img['id'] == idx:
                        file_name = img['file_name']
                        line = idx.split('-')[0] + '/' + file_name + '.jpg;'
                        break
                ymin, xmin, width, height = anno['bbox']  # the annotation xmin, ymin, xmax, ymax is not correct.
                xmax = xmin + width
                ymax = ymin + height
                line = line + str(int(xmin))+' '+str(int(ymin))+' '+str(int(xmax))+' '+str(int(ymax))+' '
                if line[-1] != ';':
                    fbox.write(line[:-1] + '\n')


'''
small polyp dataset
'''
def mask_to_box():
    data_path = '/mntnfs/med_data5/lizhuo/polyp_data/overall_images/'
    mode      = 'test'
    with open(data_path+'/test_box.txt', 'w') as fbox:
        imgs = os.listdir(data_path+mode+'/images')
        for img in tqdm(sorted(imgs)):
            # print(img)

            image = cv2.imread(data_path+mode+'/images/'+img)
            H,W,C = image.shape
 
            line = data_path+mode+'/images/'+img+';'

            mask = cv2.imread(data_path+mode+'/mask/'+img.replace('.jpg', '.png'), cv2.IMREAD_GRAYSCALE)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                xmin    = max(int(x), 0)
                ymin    = max(int(y), 0)
                xmax    = min(int(x+w), W-1)
                ymax    = min(int(y+h), H-1)
                line    = line+str(int(xmin))+' '+str(int(ymin))+' '+str(int(xmax))+' '+str(int(ymax))+' '
            if line[-1]!=';':
                fbox.write(line[:-1]+'\n')


if __name__=='__main__':
    # anno2coco_sch('/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp/Labeled_test')
    # anno2coco_sysu('/mntnfs/med_data4/yuncheng/DATASET/SCH_ZSPolyp/train')
    # anno2coco_sunseg()
    # save_gt_sch('/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp/Labeled_test')
    # save_gt_sysu('/mntnfs/med_data4/yuncheng/DATASET/ZSPolyp/test')
    # save_gt_sun('/mntnfs/med_data4/yuncheng/DATASET/SUN-SEG')
    # show_anno('./test')
    # name_list('./train')
    # name_list('./test')
    # save_color('./train')
    # save_box('/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp/Labeled_test')
    # show()
    # save_box()
    # save_lst_sch('/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp/Labeled_test')
    # save_lst_sysu('/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp')
    # save_lst_sun('/mntnfs/med_data4/yuncheng/DATASET/SUN-SEG')
    # STFTFormat('/mntnfs/med_data4/yuncheng/DATASET/ZSPolyp')
    # mask_to_box()
    anno2coco_smallpolyp('/mntnfs/med_data5/lizhuo/polyp_data/overall_images/test')
