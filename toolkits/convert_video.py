import os
import imageio
import cv2
from tqdm import tqdm

def create_gif(path):
    frames = []
    for name in tqdm(sorted(os.listdir(path))):
        image = imageio.imread(path+'/'+name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
    imageio.mimsave(path.split('/')[-1]+'.mp4', frames, 'MP4')
    return

if __name__=='__main__':
    create_gif('/mntnfs/med_data4/yiwenhu/SCHPolyp/inference/Centernet_SCHSZ/47')