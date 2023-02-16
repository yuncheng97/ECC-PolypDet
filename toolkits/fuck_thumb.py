import numpy as np
import os


def main():
    path = '/mntnfs/med_data5/yuncheng/DATASET/SCHPolyp/train/'
    for folder in os.listdir(path):
        for item in os.listdir(path + folder):
            if '.DS_' in item:
                print('fuck you', item)
                os.remove(path+folder+'/'+item)

if __name__ == '__main__':
    main()