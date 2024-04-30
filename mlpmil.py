import torch
from torch.cuda.amp import GradScaler
from torch.cuda import max_memory_allocated
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import glob
from torch.utils.data import DataLoader
import random
import os
import time
from transformers import get_cosine_schedule_with_warmup
from utils import train_MIL, evaluate, inference, MyDataset, MyLoss, read_files
from models import MODEL


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
RA_files = glob.glob('/xiongjun/test/MIL/data/0.9_RA/*/*.csv')
print(len(RA_files))

# 获取所有的标签
labels = set([file.split('/')[-2] for file in RA_files])
print(labels)

# 初始化训练集、验证集和测试集
train_files = []
val_files = []
test_files = []

# 对每个标签，随机分配文件到训练集、验证集和测试集
for label in labels:
    label_files = [file for file in RA_files if file.split('/')[-2] == label]
    np.random.shuffle(label_files)
    n = len(label_files)
    train_files += label_files[:int(n*1.0)]
    val_files += label_files[int(n*1.0):int(n*1.0)]
    test_files += label_files[int(n*1.0):]

print(f"Train: {len(train_files)} Val: {len(val_files)} Test: {len(test_files)}")

train_dataset = MyDataset(read_files(train_files))
# val_dataset = MyDataset(read_files(val_files))
# test_dataset = MyDataset(read_files(test_files))

train_loss = []
test_acc = []
test_loss = []
train_acc = []
val_acc = []
val_loss = []
lr_decay = []

mode = "4mer"
topk = 40
best_acc = 0
best_model = ""
start_time = time.time()
hidden_size =128
input_size = 20

model = MODEL(input_size, hidden_size, len(labels)).to(device)

num_epochs = 10
lr = 0.01
batch_size1 = 512
batch_size2 = 512
save_dir = "/xiongjun/test/MIL/new_saved"
file_dir = "{}_{}_{}".format(topk, hidden_size, mode)
file_name = "MIL"

criterion = MyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scaler = GradScaler()



lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, \
                                    num_training_steps=len(train_files) * topk / batch_size2 * num_epochs)

for epoch in range(num_epochs): 
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_dataset.extract_data()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size1, shuffle=False)
    topk_list = inference(model, train_dataloader, device, topk, mode)
    train_dataset.extract_data(topk_list)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size2, shuffle=True)
    train_loss_, train_acc_ = train_MIL(model, train_dataloader, criterion, device, lr_decay, optimizer, scaler, lr_scheduler, mode)
    train_acc.append(train_acc_)
    train_loss.append(train_loss_)

    # val_dataset.extract_data()
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size1, shuffle=False)
    # topk_list = inference(model, val_dataloader, device, 1, mode)
    # val_acc_ = evaluate(topk_list)

    # val_acc.append(val_acc_)
    print("Acc:", train_acc_)
    if train_acc_ > best_acc:
        if os.path.exists(os.path.join(save_dir, file_name, file_dir)) is False:
            os.makedirs(os.path.join(save_dir, file_name, file_dir))
        best_acc = train_acc_
        best_model = os.path.join(os.path.join(save_dir, file_name, file_dir), "{}-{}-{}.pth".format(file_name, epoch, best_acc))
        torch.save(model.state_dict(), best_model)
    memory = max_memory_allocated()
    print('memory allocated:',memory/(1024 ** 3), 'G')

end_time = time.time()
duration = int(end_time - start_time)
print("duration time: {} s".format(duration))

# Plot the training and validation loss accuracy and learning rate
fig, axes = plt.subplots(1, 3)
axes[0].plot(list(range(1, num_epochs + 1)), train_loss, color="r", label="train loss")
axes[0].legend()
axes[0].set_title("Loss")

axes[1].plot(list(range(1, num_epochs + 1)), train_acc, color="r", label="train acc")
#axes[1].plot(list(range(1, num_epochs + 1)), val_acc, color="b", label="val acc")
axes[1].legend()
axes[1].set_title("Accuracy")

axes[2].plot(list(range(1, len(lr_decay) + 1)), lr_decay, color="r", label="lr")
axes[2].legend()
axes[2].set_title("Learning Rate")

plt.suptitle('memory: {} G , duration: {} s'.format(memory / 1e9, duration))
plt.savefig(os.path.join(save_dir, 'figs', "{}.jpg".format(file_name)))
plt.show()