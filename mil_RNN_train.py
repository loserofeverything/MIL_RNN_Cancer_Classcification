import os
import time
import argparse
import re
import ast
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.optim as optim
from utils import train_RNN, test_RNN, rnndataset, read_files, inference, MyLoss, Preparation
from models import RNN

def main(args):
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
    train_acc = []
    val_acc = []
    val_loss = []


    best_acc = 0
    best_model = ""
    start_time = time.time()

    args.topk = int(args.MIL_dir.split("_")[0])
    args.input_size = int(args.MIL_dir.split("_")[1])
    args.mode = args.MIL_dir.split("_")[2]
    # rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=2, batch_first=True).to(device)
    rnn = RNN(args.input_size, args.RNN_hidden_size, args.RNN_num_layers, 7, device)
    criterion = MyLoss()
    optimizer = optim.SGD(rnn.parameters(), 0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)

    args.RNN_dir = args.MIL_dir + "_" + str(args.RNN_hidden_size)\
          + "_" + str(args.RNN_num_layers)
    prepare = Preparation(args.save_dir, args.MIL_dir, args.RA_path,\
        device = device,ratio = ast.literal_eval(args.split_ratio))

    # load or make dataset
    train_dataset, val_dataset, test_dataset = prepare.loadData()

    #load model
    model = prepare.loadModel(20, 128, 7)
    embs = model.emb
    # 固定模型参数
    for param in model.parameters():
        param.requires_grad = False

    #get dataloader and save topk data
    train_dataloader = prepare.extractDataLoader(train_dataset, model, args.batch_size, args.topk, "train")
    val_dataloader = prepare.extractDataLoader(val_dataset, model, args.batch_size, args.topk, "val")
    test_dataloader = prepare.extractDataLoader(test_dataset, model, args.batch_size, args.topk, "test")

    """# 获取所有的文件
    RA_files = glob.glob('/xiongjun/test/MIL/share/RA2/*/*.csv')
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
        train_files += label_files[:int(n*0.7)]
        val_files += label_files[int(n*0.7):int(n*0.9)]
        test_files += label_files[int(n*0.9):]

    print(f"Train: {len(train_files)} Val: {len(val_files)} Test: {len(test_files)}")

    train_dataset = rnndataset(read_files(train_files))
    val_dataset = rnndataset(read_files(val_files))
    test_dataset = rnndataset(read_files(test_files))"""

    """# 读取存档文件夹下准确率最高的模型文件
    save_dir = "/xiongjun/test/MIL/new_saved"
    file_name = 'MIL'
    # 获取目录下的所有文件名
    files = os.listdir(os.path.join(save_dir, file_name))
    pattern = re.compile(r'-([\d\.]+)\.pth$')
    matches = [(f, float(pattern.search(f).group(1))) for f in files if pattern.search(f)]

    save_name = 'MIL-RNN'

    # 找出准确率最高的文件名
    best_model = max(matches, key=lambda x: x[1])[0]
    model = MODEL(20, 128, 7).to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, file_name, best_model)))

    embs = model.emb"""



    """# 固定模型参数
    for param in model.parameters():
        param.requires_grad = False



    train_loss = []
    test_acc = []
    test_loss = []
    train_acc = []
    val_acc = []
    val_loss = []

    topk = 20
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




    topk_save = os.path.join(save_dir,'topk_data')
    if os.path.exists(topk_save) is False:
        os.makedirs(topk_save)

    train_dataset.setmode(0)
    train_dataset.extract_data()
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    topk_list = inference(model, train_dataloader, device, topk)
    train_dataset.setmode(1)
    train_dataset.extract_data(k = topk, topk = topk_list)
    train_dataset.df_extract.to_csv(os.path.join(topk_save, "train.csv"), index=False)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)""


    val_dataset.setmode(0)
    val_dataset.extract_data()
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    topk_list = inference(model, val_dataloader, device, topk)
    val_dataset.setmode(1)
    val_dataset.extract_data(k = topk, topk = topk_list)
    val_dataset.df_extract.to_csv(os.path.join(topk_save, "val.csv"), index=False)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset.setmode(0)
    test_dataset.extract_data()
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    topk_list = inference(model, test_dataloader, device, topk)
    test_dataset.setmode(1)
    test_dataset.extract_data(k = topk, topk = topk_list)
    test_dataset.df_extract.to_csv(os.path.join(topk_save, "test.csv"), index=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)"""

    for epoch in range(args.epochs):
        train_loss_, train_acc_ = train_RNN(epoch, embs, rnn, train_dataloader, criterion, optimizer, device)
        train_acc.append(train_acc_)
        train_loss.append(train_loss_)
        val_loss_, val_acc_ = test_RNN(epoch, embs, rnn, val_dataloader, criterion, device,\
                                picname="val_epoch{}".format(epoch))
        val_acc.append(val_acc_)
        val_loss.append(val_loss_)
        print("Val Acc:", val_acc_)
        if val_acc_ > best_acc:
            if os.path.exists(os.path.join(args.save_dir, 'RNN', args.RNN_dir)) is False:
                os.makedirs(os.path.join(args.save_dir, 'RNN', args.RNN_dir))
            best_acc = val_acc_
            best_model = os.path.join(os.path.join(args.save_dir, 'RNN', args.RNN_dir), "RNN-{}-{}.pth".format(epoch, best_acc))
            torch.save(rnn.state_dict(), best_model)



    # 获取目录下的所有文件名
    files = os.listdir(os.path.join(args.save_dir, 'RNN', args.RNN_dir))
    pattern = re.compile(r'-([\d\.]+)\.pth$')
    matches = [(f, float(pattern.search(f).group(1))) for f in files if pattern.search(f)]
    # 找出准确率最高的文件名
    best_model = max(matches, key=lambda x: x[1])[0]
    new_rnn = RNN(args.input_size, args.RNN_hidden_size, args.RNN_num_layers, 7, device)
    new_rnn.load_state_dict(torch.load(os.path.join(args.save_dir, 'RNN',args.RNN_dir, best_model)))

    test_loss_, test_acc_ = test_RNN(114514, embs, new_rnn, test_dataloader, \
            criterion, device, picname=best_model.split('.')[0] + "_test")

    end_time = time.time()
    duration = int(end_time - start_time)
    print("duration time: {} s".format(duration))    
    print(test_acc_, test_loss_)
    # Plot the training and validation loss accuracy and learning rate
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(list(range(1, args.epochs + 1)), train_loss, color="r", label="train loss")
    axes[0].plot(list(range(1, args.epochs + 1)), val_loss, color="b", label="val loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(list(range(1, args.epochs + 1)), train_acc, color="r", label="train acc")
    axes[1].plot(list(range(1, args.epochs + 1)), val_acc, color="b", label="val acc")
    axes[1].legend()
    axes[1].set_title("Accuracy")


    plt.suptitle('duration: {} s'.format(duration))
    plt.savefig(os.path.join(args.save_dir, 'RNN',args.RNN_dir, "{}.jpg".format("acc_loss")))
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MIL-RNN-train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--MIL_dir", type=str, default="40_128_4mer", help="MIL Model file folder should be saved in save_dir")
    parser.add_argument("--split_ratio", type=str, default="[0.7,0.2,0.1]", help="Split ratio of train, val and test")
    parser.add_argument("--RNN_hidden_size", type=int, default=256, help="RNN hidden size")
    parser.add_argument("--RNN_num_layers", type=int, default=2, help="RNN num layers")
    parser.add_argument("--save_dir", type=str, default="/xiongjun/test/MIL/new_saved", help="Save directory")
    parser.add_argument("--RA_path", type=str, default="/xiongjun/test/MIL/data/0.9_RA/*/*.csv", help="RA path")
    parser.add_argument("--save_figs", type=str, default="/xiongjun/test/MIL/figs", help="Save ROC figures")
    args = parser.parse_args()
    main(args)






