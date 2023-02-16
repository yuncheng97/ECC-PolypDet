import os, sys, shutil
from tqdm import tqdm

# /mntnfs/med_data4/yuncheng/DATASET/SCHPolyp

def merge_data():
    source_folder = '/mntnfs/med_data4/yuncheng/DATASET/ZSPolyp/test'
    target_folder = '/mntnfs/med_data4/yuncheng/DATASET/SYSU/Data'
    folds = os.listdir(source_folder)
    # folds.remove('.DS_Store')
    # vclips = sorted(folds, key=lambda x: int(x))

    for idx in folds:
        OldFolder = os.path.join(source_folder, idx)
        print(OldFolder)
        # newidx = str(int(idx) + 300)
        NewFolder = os.path.join(target_folder, idx)
        print(NewFolder)
        shutil.copytree(OldFolder, NewFolder)


if __name__ == "__main__":
    merge_data()