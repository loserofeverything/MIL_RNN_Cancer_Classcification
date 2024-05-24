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
import time
from transformers import get_cosine_schedule_with_warmup
from utils import train_GATTNET, test_GATTNET, rnndataset, BGATTNET_Loss, read_files
from models import MODEL, BGAttNet


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
    train_files += label_files[:int(n * 0.8)]
    val_files += label_files[int(n * 0.8):int(n * 1.0)]
    test_files += label_files[int(n * 1.0):]

print(f"Train: {len(train_files)} Val: {len(val_files)} Test: {len(test_files)}")

# with open(\
#     "/xiongjun/test/MIL/new_saved/dataset/0.9_RA_[0.7-0.2-0.1]/train.pkl", 'rb')\
#           as f:
#     train_dataset = pickle.load(f)
# with open(\
#     "/xiongjun/test/MIL/new_saved/dataset/0.9_RA_[0.7-0.2-0.1]/val.pkl", 'rb')\
#           as f:
#     val_dataset = pickle.load(f)
# with open(\
#     "/xiongjun/test/MIL/new_saved/dataset/0.9_RA_[0.7-0.2-0.1]/test.pkl", 'rb')\
#           as f:
#     test_dataset = pickle.load(f)


if os.path.exists("/xiongjun/test/MIL/new_saved/dataset/top10k_sampleBatchDataset") is False:
     os.mkdir("/xiongjun/test/MIL/new_saved/dataset/top10k_sampleBatchDataset")


if os.path.isfile("/xiongjun/test/MIL/new_saved/dataset/top10k_sampleBatchDataset/train.pkl"):
     with open("/xiongjun/test/MIL/new_saved/dataset/top10k_sampleBatchDataset/train.pkl", 'rb') as f:
          train_dataset = pickle.load(f)
else:    
    train_dataset = rnndataset(read_files(train_files))
    with open(os.path.join("/xiongjun/test/MIL/new_saved/dataset/top10k_sampleBatchDataset", \
                       'train.pkl'), 'wb') as f:
            pickle.dump(train_dataset, f)

if os.path.isfile("/xiongjun/test/MIL/new_saved/dataset/top10k_sampleBatchDataset/val.pkl"):
     with open("/xiongjun/test/MIL/new_saved/dataset/top10k_sampleBatchDataset/val.pkl", 'rb') as f:
          val_dataset = pickle.load(f)
else:    
    val_dataset = rnndataset(read_files(val_files))
    with open(os.path.join("/xiongjun/test/MIL/new_saved/dataset/top10k_sampleBatchDataset", \
                       'val.pkl'), 'wb') as f:
            pickle.dump(val_dataset, f)

# if os.path.isfile("/xiongjun/test/MIL/new_saved/dataset/top10k_sampleBatchDataset/test.pkl"):
#      with open("/xiongjun/test/MIL/new_saved/dataset/top10k_sampleBatchDataset/test.pkl", 'rb') as f:
#           test_dataset = pickle.load(f)
# else:    
#     test_dataset = rnndataset(read_files(test_files))
#     with open(os.path.join("/xiongjun/test/MIL/new_saved/dataset/top10k_sampleBatchDataset", \
#                        'test.pkl'), 'wb') as f:
#             pickle.dump(test_dataset, f)

train_dataset.setmode(1)
train_dataset.extract_data()

val_dataset.setmode(1)
val_dataset.extract_data()

# test_dataset.setmode(1)
# test_dataset.extract_data()

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
instance_embedding_dim =128
num_instance_concepts = 16
bag_embedding_dim = 256
input_dim = 20
num_class = len(labels)

model = BGAttNet(input_dim, instance_embedding_dim, num_instance_concepts, \
                bag_embedding_dim, num_class).to(device)

num_epochs = 100
lr = 1e-4
batch_size = 8
save_dir = "/xiongjun/test/MIL/new_saved"
file_dir = "ver1"
file_name = "gattnet"

criterion = BGATTNET_Loss(0.05)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scaler = GradScaler()


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=540)

for epoch in range(num_epochs): 
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss_, train_acc_ = train_GATTNET(model, train_dataloader, criterion, device, optimizer, scaler, mode)
    train_acc.append(train_acc_)
    train_loss.append(train_loss_)
    print("epoch-{}: trian Acc{}:".format(epoch + 1, train_acc_))
    # val_dataset.setmode(0)
    # val_dataset.extract_data()
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size1, shuffle=False)
    # topk_list = inference(model, val_dataloader, device, 1, mode)
    # val_acc_ = evaluate(topk_list)

    val_loss_, val_acc_, attn_dicts, result_dicts = test_GATTNET(model, val_dataloader, criterion, epoch, device, mode=mode)
    val_acc.append(val_acc_)
    print("epoch-{}: val Acc{}:".format(epoch + 1, val_acc_))
    if True:
        if os.path.exists(os.path.join(save_dir, file_name, file_dir)) is False:
            os.makedirs(os.path.join(save_dir, file_name, file_dir))
        best_acc = train_acc_
        best_model = os.path.join(os.path.join(save_dir, file_name, file_dir), "{}-{}-{}.pth".format(file_name, epoch, best_acc))
        torch.save(model.state_dict(), best_model)
#     # if os.path.exists(os.path.join(save_dir, file_name, file_dir)) is False:
#     #     os.makedirs(os.path.join(save_dir, file_name, file_dir))
#     # best_acc = train_acc_
#     # best_model = os.path.join(os.path.join(save_dir, file_name, file_dir), "{}-{}-{}.pth".format(file_name, epoch, best_acc))
#     # torch.save(model.state_dict(), best_model)
    memory = max_memory_allocated()
    print('memory allocated:',memory/(1024 ** 3), 'G')

with open("/xiongjun/test/MIL/new_saved/gattnet/top40_instances/result.pkl","wb") as f:
    pickle.dump(result_dicts, f)
with open("/xiongjun/test/MIL/new_saved/gattnet/top40_instances/attn.pkl","wb") as f:
    pickle.dump(attn_dicts, f)

end_time = time.time()
duration = int(end_time - start_time)
print("duration time: {} s".format(duration))

# Plot the training and validation loss accuracy and learning rate
fig, axes = plt.subplots(1, 2)
axes[0].plot(list(range(1, num_epochs + 1)), train_loss, color="r", label="train loss")
axes[0].plot(list(range(1, num_epochs + 1)), val_loss, color="r", label="val loss")
axes[0].legend()
axes[0].set_title("Loss")

axes[1].plot(list(range(1, num_epochs + 1)), train_acc, color="r", label="train acc")
axes[1].plot(list(range(1, num_epochs + 1)), val_acc, color="b", label="val acc")
axes[1].legend()
axes[1].set_title("Accuracy")


plt.suptitle('memory: {} G , duration: {} s'.format(memory / (1024 ** 3), duration))
plt.savefig(os.path.join(save_dir, 'figs', "{}.jpg".format(file_name)))
plt.show()