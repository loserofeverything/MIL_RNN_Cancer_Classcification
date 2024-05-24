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
from utils import conv_train_tcr_RNN, conv_test_tcr_RNN, MyLoss, tcrPreparation
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
    args.RNN_num_layers = 1
    args.topk = 1
    args.mode = "tcr"
    # rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=2, batch_first=True).to(device)
    rnn = RNN(args.input_size, args.RNN_hidden_size, args.RNN_num_layers, 2, device)
    criterion = MyLoss()
    optimizer = optim.SGD(rnn.parameters(), 0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)

    args.RNN_dir = args.MIL_dir + "_" + str(args.RNN_hidden_size)\
          + "_" + str(args.RNN_num_layers) + "_top{}".format(args.topk)
    prepare = tcrPreparation(args.save_dir, args.MIL_dir, args.RA_path,\
        device = device,ratio = ast.literal_eval(args.split_ratio), \
            model_type_dir=args.model_type_dir)

    # load or make dataset
    train_dataset, val_dataset = prepare.loadData(is_test = False, unique_dataset_name="split_label_0_6")
    train_dataset.df['label'] = (train_dataset.df['label'] == 0).astype(int)
    val_dataset.df['label'] = (val_dataset.df['label'] == 0).astype(int)
    max_length = train_dataset.max_length
    #load model
    model = prepare.loadModel(80, 128, 2)
    embs = model.emb
    # 固定模型参数
    for param in model.parameters():
        param.requires_grad = False

    #get dataloader and save topk data
    train_dataloader = prepare.extractDataLoader(train_dataset, model, args.batch_size, args.topk, "train", unique_dataset_name='split_label_0_6_[0.8-0.2]')
    val_dataloader = prepare.extractDataLoader(val_dataset, model, args.batch_size, args.topk, "val", unique_dataset_name='split_label_0_6_[0.8-0.2]')
    

    

    for epoch in range(args.epochs):
        train_loss_, train_acc_ = conv_train_tcr_RNN(epoch, embs, rnn, train_dataloader, criterion, optimizer, device, max_length)
        train_acc.append(train_acc_)
        train_loss.append(train_loss_)
        val_loss_, val_acc_ = conv_test_tcr_RNN(epoch, embs, rnn, val_dataloader, criterion, device,\
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
        elif (epoch+1)%10 == 0:
            if os.path.exists(os.path.join(args.save_dir, 'RNN', args.RNN_dir)) is False:
                os.makedirs(os.path.join(args.save_dir, 'RNN', args.RNN_dir))
            saved_model = os.path.join(os.path.join(args.save_dir, 'RNN', args.RNN_dir), "RNN-{}-{}.pth".format(epoch, val_acc_))
            torch.save(model.state_dict(), saved_model)



    end_time = time.time()
    duration = int(end_time - start_time)
    print("duration time: {} s".format(duration))    
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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--MIL_dir", type=str, default="top1_fixed_dim_0_6", help="save dir of model with specific params set")
    parser.add_argument("--split_ratio", type=str, default="[0.7,0.2,0.1]", help="Split ratio of train, val and test")
    parser.add_argument("--RNN_hidden_size", type=int, default=256, help="RNN hidden size")
    parser.add_argument("--RNN_num_layers", type=int, default=2, help="RNN num layers")
    parser.add_argument("--save_dir", type=str, default="/xiongjun/test/MIL/tcr_saved", help="Save directory")
    parser.add_argument("--RA_path", type=str, default="/xiongjun/test/MIL/data/0.9_tcr/*/*.csv", help="RA path")
    parser.add_argument("--save_figs", type=str, default="/xiongjun/test/MIL/tcr_saved/figs/RNN_top1_conv1d_figs_0_6", help="Save ROC figures")
    parser.add_argument("--topk", type=int, default=10, help="topk tcr to train")
    parser.add_argument("--input_size", type=int, default=128, help="input size of RNN")
    parser.add_argument("--model_type_dir", type=str, default="conv1dMIL", help="root dir of a specific model type")
    
    args = parser.parse_args()
    main(args)






