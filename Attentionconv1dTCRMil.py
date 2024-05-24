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
from models import MODEL, AttentionMIL, LocalAttentionMIL




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


# # 获取所有的文件
# RA_files = glob.glob('/xiongjun/test/MIL/data/0.9_tcr/*/*.csv')
# print(len(RA_files))

# # 获取所有的标签
# labels = set([file.split('/')[-2] for file in RA_files])
# print(labels)

# # 初始化训练集、验证集和测试集
# train_files = []
# val_files = []
# test_files = []

# # 对每个标签，随机分配文件到训练集、验证集和测试集
# for label in labels:
#     label_files = [file for file in RA_files if file.split('/')[-2] == label]
#     np.random.shuffle(label_files)
#     n = len(label_files)
#     train_files += label_files[:int(n * 0.8)]
#     val_files += label_files[int(n * 0.8):int(n * 1.0)]
#     test_files += label_files[int(n * 1.0):]

# print(f"Train: {len(train_files)} Val: {len(val_files)} Test: {len(test_files)}")

with open("/xiongjun/test/MIL/tcr_saved/dataset/0.9_tcr_[0.7-0.2-0.1]/train.pkl","rb") as f:
    train_dataset = pickle.load(f)
with open("/xiongjun/test/MIL/tcr_saved/dataset/0.9_tcr_[0.7-0.2-0.1]/val.pkl","rb") as f:
    val_dataset = pickle.load(f)
with open("/xiongjun/test/MIL/tcr_saved/dataset/0.9_tcr_[0.7-0.2-0.1]/test.pkl","rb") as f:
    test_dataset = pickle.load(f)


# 




# print("max length:", max_length)

# train_dataset.padding()
train_dataset.setmode(1)
val_dataset.setmode(1)
test_dataset.setmode(1)

# val_dataset.padding()
# val_dataset.setmode(0)

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
topk = 1
best_acc = 0
best_model = ""
start_time = time.time()
hidden_dim = 128
num_classes = 7


# model = MODEL(input_dim, hidden_dim, num_class).to(device)


num_epochs = 100
lr = 0.005
radius = 16
batch_size = 1
save_dir = "/xiongjun/test/MIL/tcr_saved"
file_dir = "batch_size{}_radius{}_hidden_dim{}".format(batch_size, radius, hidden_dim)
file_name = "LocalAttentionconv1dMIL"

model = LocalAttentionMIL(hidden_dim, num_classes, local_radius=radius).to(device)

criterion = MyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scaler = GradScaler()


# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=540)

train_dataset.extract_data()
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
val_dataset.extract_data()
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

max_length = 16

for epoch in range(num_epochs): 
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss_, train_acc_ = attention_conv1d_tcr_train_MIL(model, train_dataloader, criterion, device, optimizer, scaler)
    train_acc.append(train_acc_)
    train_loss.append(train_loss_)
    print("epoch-{}: trian Acc{}:".format(epoch + 1, train_acc_))
    
    
    val_loss_, val_acc_, attns_dict = attention_conv1d_tcr_test_MIL(model, val_dataloader, criterion, device, \
                    picname='val_epoch{}'.format(epoch), save_dir = '/xiongjun/test/MIL/tcr_saved/figs/LocalAttention_conv1d_figs', mode = 'val')

    val_acc.append(val_acc_)
    val_loss.append(val_loss_)

    if val_acc_ > best_acc:
        if os.path.exists(os.path.join(save_dir, file_name, file_dir)) is False:
            os.makedirs(os.path.join(save_dir, file_name, file_dir))
        best_acc = val_acc_
        best_model = os.path.join(os.path.join(save_dir, file_name, file_dir), "{}-{}-{}.pth".format(file_name, epoch, best_acc))
        torch.save(model.state_dict(), best_model)

    elif (epoch+1)%10 == 0:
        if os.path.exists(os.path.join(save_dir, file_name, file_dir)) is False:
            os.makedirs(os.path.join(save_dir, file_name, file_dir))
        saved_model = os.path.join(os.path.join(save_dir, file_name, file_dir), "{}-{}-{}.pth".format(file_name, epoch, val_acc_))
        torch.save(model.state_dict(), saved_model)

#     # if os.path.exists(os.path.join(save_dir, file_name, file_dir)) is False:
#     #     os.makedirs(os.path.join(save_dir, file_name, file_dir))
#     # best_acc = train_acc_
#     # best_model = os.path.join(os.path.join(save_dir, file_name, file_dir), "{}-{}-{}.pth".format(file_name, epoch, best_acc))
#     # torch.save(model.state_dict(), best_model)
    memory = max_memory_allocated()
    print('memory allocated:',memory/(1024 ** 3), 'G')




test_dataset.extract_data()
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
# 获取目录下的所有文件名
files = os.listdir(os.path.join(save_dir, file_name, file_dir))
pattern = re.compile(r'-([\d\.]+)\.pth$')
matches = [(f, float(pattern.search(f).group(1))) for f in files if pattern.search(f)]
# 找出准确率最高的文件名
best_model = max(matches, key=lambda x: x[1])[0]
loadmodel = LocalAttentionMIL(hidden_dim, num_classes, local_radius=radius).to(device)
loadmodel.load_state_dict(torch.load(os.path.join(save_dir, file_name,file_dir, best_model)))
test_loss_, test_acc_, attns_dict = attention_conv1d_tcr_test_MIL(model, test_dataloader, criterion, device, \
                    picname=best_model.split('.')[0] + "_test", save_dir = '/xiongjun/test/MIL/tcr_saved/figs/LocalAttention_conv1d_figs')


if os.path.exists(os.path.join(save_dir, file_name, file_dir, 'attn_dicts')) is False:
    os.makedirs(os.path.join(save_dir, file_name, file_dir, 'attn_dicts'))

with open(os.path.join(save_dir, file_name, file_dir, 'attn_dicts', 'result.pkl'),"wb") as f:
    pickle.dump(attns_dict, f)

print("test acc:{}, test loss:{}".format(test_acc_, test_loss_))

end_time = time.time()
duration = int(end_time - start_time)
print("duration time: {} s".format(duration))

# Plot the training and validation loss accuracy and learning rate
fig, axes = plt.subplots(1, 2)
axes[0].plot(list(range(1, num_epochs + 1)), train_loss, color="r", label="train loss")
axes[0].plot(list(range(1, num_epochs + 1)), val_loss, color="b", label="val loss")
axes[0].legend()
axes[0].set_title("Loss")

axes[1].plot(list(range(1, num_epochs + 1)), train_acc, color="r", label="train acc")
axes[1].plot(list(range(1, num_epochs + 1)), val_acc, color="b", label="val acc")
axes[1].legend()
axes[1].set_title("Accuracy")


plt.suptitle('memory: {} G , duration: {} s'.format(memory / (1024 ** 3), duration))
plt.savefig(os.path.join(save_dir, file_name, "{}.jpg".format(file_dir)))
plt.show()