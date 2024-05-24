import torch
import pickle
from torch.cuda.amp import GradScaler
from torch.cuda import max_memory_allocated
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import glob
from torch.utils.data import DataLoader
import random
import os
import re
import time
from utils import attention_conv1d_tcr_train_MIL, attention_conv1d_tcr_test_MIL, \
    tcr_dataset, MyLoss, read_tcr_files
from models import MODEL, AttentionMIL




# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if device.type == "cuda":
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
random.seed(seed)

# Set cuDNN configurations for determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 获取所有的文件
RA_files = glob.glob('/xiongjun/test/MIL/data/0.9_tcr/*/*.csv')
print(len(RA_files))

chosen = ['1','6']
# 获取所有的标签
# labels = set([file.split('/')[-2] for file in RA_files])
labels = set(chosen)
print(labels)

# 初始化训练集、验证集和测试集
train_files = []
val_files = []
test_files = []

file_dir = "/xiongjun/test/MIL/tcr_saved/dataset"
file_name = "split_label_{}_{}".format(chosen[0], chosen[1])
file_path = os.path.join(file_dir, file_name)

# 对每个标签，随机分配文件到训练集、验证集和测试集
for label in labels:
    label_files = [file for file in RA_files if file.split('/')[-2] == label]
    np.random.shuffle(label_files)
    n = len(label_files)
    train_files += label_files[:int(n * 0.8)]
    val_files += label_files[int(n * 0.8):int(n * 1.0)]

print(f"Train: {len(train_files)} Val: {len(val_files)}")



if os.path.exists(file_path) is False:
     os.mkdir(file_path)


if os.path.isfile(os.path.join(file_path, 'train.pkl')) is False:

    train_dataset = tcr_dataset(read_tcr_files(train_files))
    train_dataset.max_length = 16
    train_dataset.padding()
    with open(os.path.join(file_path, 'train.pkl'),'wb') as f:
            pickle.dump(train_dataset, f)

if os.path.isfile(os.path.join(file_path, 'val.pkl')) is False:   
    
    val_dataset = tcr_dataset(read_tcr_files(val_files))
    val_dataset.max_length = 16
    val_dataset.padding()
    with open(os.path.join(file_path, 'val.pkl'),'wb') as f:
            pickle.dump(val_dataset, f)