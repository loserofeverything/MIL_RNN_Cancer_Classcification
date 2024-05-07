import os
import time
import re
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.optim as optim
from utils import train_RNN, test_RNN, MyLoss, Preparation
from models import RNN


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

train_loss = []
test_acc = []
test_loss = []
train_acc = []
val_acc = []
val_loss = []

topk = 40
best_acc = 0
best_model = ""
start_time = time.time()

mode = "4mer"
epochs = 10
batch_size = 128
batch_size2 = 1
# rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=2, batch_first=True).to(device)
rnn = RNN(128, 256, 2, 7, device)
criterion = MyLoss()
optimizer = optim.SGD(rnn.parameters(), 0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)

save_name = 'MIL-RNN'
save_dir = "/xiongjun/test/MIL/saved"
RA_path = '/xiongjun/test/MIL/data/RA/*/*.csv'


prepare = Preparation(save_dir, 'MIL', RA_path, device, ratio=[0.7, 0.2, 0.1])

# load or make dataset
train_dataset, val_dataset, test_dataset = prepare.loadData()

#load model
model = prepare.loadModel(20, 128, 7)
embs = model.emb
# 固定模型参数
for param in model.parameters():
    param.requires_grad = False

#get dataloader and save topk data
train_dataloader = prepare.extractDataLoader(train_dataset, model, batch_size, topk, "train")
val_dataloader = prepare.extractDataLoader(val_dataset, model, batch_size, topk, "val")
test_dataloader = prepare.extractDataLoader(test_dataset, model, batch_size, topk, "test")

# 获取目录下的所有文件名
files = os.listdir(os.path.join(save_dir, save_name))
pattern = re.compile(r'-([\d\.]+)\.pth$')
matches = [(f, float(pattern.search(f).group(1))) for f in files if pattern.search(f)]
# 找出准确率最高的文件名
best_model = max(matches, key=lambda x: x[1])[0]
new_rnn = RNN(128, 256, 2, 7, device)
new_rnn.load_state_dict(torch.load(os.path.join(save_dir, save_name, best_model)))
_loss_, _acc_ = test_RNN(114514, embs, new_rnn, train_dataloader, criterion, device)
print(_acc_, _loss_)