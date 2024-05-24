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
from utils import train_tcr_RNN, test_tcr_RNN, MyLoss, tcrPreparation
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

    args.topk = 40
    args.mode = "tcr"
    # rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=2, batch_first=True).to(device)
    rnn = RNN(args.input_size, args.RNN_hidden_size, args.RNN_num_layers, 7, device)
    criterion = MyLoss()
    optimizer = optim.SGD(rnn.parameters(), 0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)

    args.RNN_dir = args.MIL_dir + "_" + str(args.RNN_hidden_size)\
          + "_" + str(args.RNN_num_layers) + "_top{}".format(args.topk)
    prepare = tcrPreparation(args.save_dir, args.MIL_dir, args.RA_path,\
        device = device,ratio = ast.literal_eval(args.split_ratio))

    # load or make dataset
    train_dataset, val_dataset, test_dataset = prepare.loadData()

    #load model
    model = prepare.loadModel(80, 128, 7)
    embs = model.emb
    # 固定模型参数
    for param in model.parameters():
        param.requires_grad = False

    #get dataloader and save topk data
    train_dataloader = prepare.extractDataLoader(train_dataset, model, args.batch_size, args.topk, "train")
    val_dataloader = prepare.extractDataLoader(val_dataset, model, args.batch_size, args.topk, "val")
    test_dataloader = prepare.extractDataLoader(test_dataset, model, args.batch_size, args.topk, "test")

 

    for epoch in range(args.epochs):
        train_loss_, train_acc_ = train_tcr_RNN(epoch, embs, rnn, train_dataloader, criterion, optimizer, device)
        train_acc.append(train_acc_)
        train_loss.append(train_loss_)
        val_loss_, val_acc_ = test_tcr_RNN(epoch, embs, rnn, val_dataloader, criterion, device,\
                                picname="val_epoch{}".format(epoch), save_dir = args.save_figs)
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

    test_loss_, test_acc_ = test_tcr_RNN(114514, embs, new_rnn, test_dataloader, \
            criterion, device, picname=best_model.split('.')[0] + "_test", save_dir = args.save_figs)

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
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--MIL_dir", type=str, default="top40_fixed_dim", help="MIL Model file folder should be saved in save_dir")
    parser.add_argument("--split_ratio", type=str, default="[0.7,0.2,0.1]", help="Split ratio of train, val and test")
    parser.add_argument("--RNN_hidden_size", type=int, default=256, help="RNN hidden size")
    parser.add_argument("--RNN_num_layers", type=int, default=2, help="RNN num layers")
    parser.add_argument("--save_dir", type=str, default="/xiongjun/test/MIL/tcr_saved", help="Save directory")
    parser.add_argument("--RA_path", type=str, default="/xiongjun/test/MIL/data/0.9_tcr/*/*.csv", help="RA path")
    parser.add_argument("--save_figs", type=str, default="/xiongjun/test/MIL/tcr_saved/figs", help="Save ROC figures")
    parser.add_argument("--topk", type=int, default=10, help="topk tcr to train")
    parser.add_argument("--input_size", type=int, default=128, help="input size of RNN")
    args = parser.parse_args()
    main(args)






