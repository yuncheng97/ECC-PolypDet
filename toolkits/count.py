from tqdm import tqdm
with open('/mntnfs/med_data4/yuncheng/DATASET/SCHPolyp/hard_cas_train_box.txt', 'r') as lines:
    hard_num = 0
    for line in tqdm(lines):
        name, box, label = line.strip().split(';')
        if label == 'hard':
            hard_num += 1
    
    print(hard_num)