import torch
import os
import numpy as np
import glob
import pickle
import re
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import pandas as pd
from itertools import groupby
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from models import RNN, MODEL
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc



class MyDataset(Dataset):
    """
    一个用于处理特定格式数据的自定义数据集类。

    Args:
        df (pandas.DataFrame): 包含数据的输入DataFrame。

    Attributes:
        df (pandas.DataFrame): 包含数据的输入DataFrame。
        df_extract (pandas.DataFrame): 提取的DataFrame。

    Methods:
        tuples_to_df(tplist): 将元组列表转换为pandas DataFrame。
        extract_data(topk): 从输入DataFrame中提取数据。

    """

    def __init__(self, df):
        self.df = df
        self.df['data'] = self.df['data'].apply(eval)
        print("df rows: ", len(self.df))

    def tuples_to_df(self, tplist):
        """
        将元组列表转换为pandas DataFrame。

        Args:
            tplist (list): 要转换的元组列表。

        Returns:
            pandas.DataFrame: 转换后的DataFrame。

        """
        rows = []
        for tup in tplist:
            # 获取元组中除了最后一个元素之外的所有元素
            row = list(tup[:-1])
            rows.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(rows, columns=['data', 'RA_4mer', 'RA_TCR', '4-mer', 'label', 'sample'])
        return df
    
    def extract_data(self, topk=None):
        """
        从输入DataFrame中提取数据。

        Args:
            topk (list of tuples, optional): 提取的前k行。默认为None。

        """
        if topk is not None:
            self.df_extract = self.tuples_to_df(topk)
        else:
            self.df_extract = self.df.copy()

    def __len__(self):
        return len(self.df_extract)

    def __getitem__(self, idx):
        data = torch.tensor(self.df_extract['data'][idx], dtype=torch.float32).view(-1)
        _4mer_RA = torch.tensor(self.df_extract['RA_4mer'][idx], dtype=torch.float32)
        _tcr_RA = torch.tensor(self.df_extract['RA_TCR'][idx], dtype=torch.float32)
        name = self.df_extract['4-mer'][idx]
        sample = self.df_extract['sample'][idx]
        label = torch.tensor(self.df_extract['label'][idx], dtype=torch.long)
        return data, _4mer_RA, _tcr_RA, name, label, sample

class rnndataset(MyDataset):
    '''
    RNN数据集类，继承自MyDataset。

    Attributes:
        mode (int): 数据集模式，0表示给embs层进行推理提供的数据，1表示给RNN层进行训练提供的数据。

    Methods:
        __len__(): 返回数据集的长度。
        setmode(mode): 设置数据集的模式。
        extract_data(k=None, topk=None): 确定要提取的数据，并在模式非0时对其分组。
        __getitem__(idx): 根据提供的索引检索项目，取决于模式。

    '''

    def __len__(self):
        '''
        返回数据集的长度。

        Returns:
            int: 数据集的长度。如果模式为0，使用基类的长度；否则，返回唯一样本的数量。
        '''
        if self.mode == 0:
            return super().__len__()
        else:
            return len(self.sample_values)

    def setmode(self, mode):
        '''
        在使用extract_data方法之前，设置数据集的模式。

        Args:
            mode (int): 数据集的模式。0表示给embs层进行推理提供的数据，1表示给RNN层进行训练提供的数据。
        '''
        self.mode = mode

    def extract_data(self, k=None, topk=None):
        '''
        确定要提取的数据，并在模式非0时对其分组。

        Args:
            k (int, optional): 如果指定，仅提取前k个元素。
            topk (list of turple, optional): 如果指定，仅提取前k个元素。

        Returns:
            None
        '''
        super().extract_data(topk)
        if self.mode == 0:
            return
        self.sample_groups = self.df_extract.groupby('sample')
        self.sample_values = self.df_extract['sample'].unique()

    def __getitem__(self, idx):
        """
        根据提供的索引检索项目，取决于模式。

        Args:
            idx (int): 要检索的元素的索引。

        Returns:
            tuple: 根据选定模式包含处理后的张量和其他数据。
                当模式为0时，返回4-mer[idx]；
                当模式为1时，返回sample[idx]的数据。
        """
        if self.mode == 0:
            return super().__getitem__(idx)
        else:
            _df = self.sample_groups.get_group(self.sample_values[idx])
            data, _4mer_RA, _tcr_RA, name, sample, label = [], [], [], [], [], []

            for idx, row in _df.iterrows():
                data.append(torch.tensor(row['data'], dtype=torch.float32).view(-1))
                _4mer_RA.append(torch.tensor(row['RA_4mer'], dtype=torch.float32))
                _tcr_RA.append(torch.tensor(row['RA_TCR'], dtype=torch.float32))
                name.append(row['4-mer'])
                sample.append(row['sample'])
                label.append(torch.tensor(row['label'], dtype=torch.long))
            data = torch.stack(data)
            _4mer_RA = torch.stack(_4mer_RA)
            _tcr_RA = torch.stack(_tcr_RA)
            label = torch.stack(label)
            return data, _4mer_RA, _tcr_RA, name, label, sample

class Preparation():
    """
    数据准备类，用于加载模型和数据集，并提供数据加载器的功能。

    Args:
        save_dir (str): 模型和数据集保存的目录路径。
        model_dir (str): 模型保存的子目录路径。
        RA_path (str): RA文件的路径。
        device (str, optional): 使用的设备，默认为"cpu"。
        dataset_dir (str, optional): 数据集保存的子目录路径，默认为"dataset"。
        topk_dir (str, optional): topk数据保存的子目录路径，默认为"topk_data"。
        ratio (list, optional): 训练集、验证集和测试集的比例，默认为[0.7, 0.2, 0.1]。

    Attributes:
        save_dir (str): 模型和数据集保存的目录路径。
        model_dir (str): 模型保存的子目录路径。
        topk_dir (str): topk数据保存的子目录路径。
        device (str): 使用的设备。
        dataset_dir (str): 数据集保存的子目录路径。
        RA_path (str): RA文件的路径。
        ratio (list): 训练集、验证集和测试集的比例。

    Methods:
        loadModel(input_dim, hidden_dim, output_dim): 加载模型。
        loadData(): 加载数据集。
        extractDataLoader(dataset, model, batch_size, topk, mode): 提取数据加载器。

    """

    def __init__(self, save_dir, model_dir, RA_path, device="cpu", \
                 dataset_dir="dataset", topk_dir="topk_data", ratio=[0.7, 0.2, 0.1]):
        self.save_dir = save_dir
        self.model_dir = model_dir
        self.topk_dir = topk_dir
        self.device = device
        self.dataset_dir = dataset_dir 
        self.RA_path = RA_path
        self.ratio = ratio

    def loadModel(self, input_dim, hidden_dim, output_dim):
        """
        加载正确率最高的模型。

        Args:
            input_dim (int): 输入维度。
            hidden_dim (int): 隐藏层维度。
            output_dim (int): 输出维度。

        Returns:
            model: 加载的模型。

        """
        model_files = os.listdir(os.path.join(self.save_dir, "MIL", self.model_dir))
        pattern = re.compile(r'-([\d\.]+)\.pth$')
        matches = [(f, float(pattern.search(f).group(1))) for f in model_files if pattern.search(f)]
        # 找出准确率最高的文件名
        best_model = max(matches, key=lambda x: x[1])[0]
        model = MODEL(input_dim, hidden_dim, output_dim).to(self.device)
        model.load_state_dict(torch.load(os.path.join(self.save_dir, 'MIL', self.model_dir, best_model)))
        return model

    def loadData(self):
        """
        加载数据集。

        Returns:
            tuple: 训练集、验证集和测试集的数据。

        """
        ra_src = self.RA_path.split('/')[-3]
        s = "[" + "-".join(map(str, self.ratio)) + "]"
        unique_dataset_name = ra_src + "_" + s
        path = os.path.join(self.save_dir, self.dataset_dir, unique_dataset_name)
        
        if os.path.exists(path) is False:
            os.mkdir(path)
        datasets = []
        file_names = ['train.pkl', 'val.pkl', 'test.pkl']
        for file_name in file_names:
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    datasets.append(data)
            else:
                break
        if len(datasets) == 3:
            return datasets[0], datasets[1], datasets[2]
        
        train_files = []
        val_files = []
        test_files = []
        RA_files = glob.glob(self.RA_path)
        labels = set([file.split('/')[-2] for file in RA_files])
        # 对每个标签，随机分配文件到训练集、验证集和测试集
        for label in labels:
            label_files = [file for file in RA_files if file.split('/')[-2] == label]
            np.random.shuffle(label_files)
            n = len(label_files)
            train_files += label_files[:int(n*self.ratio[0])]
            val_files += label_files[int(n*self.ratio[0]):int(n*(self.ratio[1] + self.ratio[0]))]
            test_files += label_files[int(n*(1.0-self.ratio[2])):]

        print(f"Train: {len(train_files)} Val: {len(val_files)} Test: {len(test_files)}")
        train_dataset = rnndataset(read_files(train_files))
        val_dataset = rnndataset(read_files(val_files))
        test_dataset = rnndataset(read_files(test_files))
        with open(os.path.join(path, 'train.pkl'), 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(os.path.join(path, 'val.pkl'), 'wb') as f:
            pickle.dump(val_dataset, f)
        with open(os.path.join(path, 'test.pkl'), 'wb') as f:
            pickle.dump(test_dataset, f)
        return train_dataset, val_dataset, test_dataset

    def extractDataLoader(self, dataset, model, batch_size, topk, mode):
        """
        提取数据加载器。保存topk的数据到文件

        Args:
            dataset: 数据集。
            model: 加载的模型。
            batch_size (int): 批处理大小。
            topk (int): top-k值。
            mode (str): 模式，可以是"train"或"test"。

        Returns:
            DataLoader: 提取的数据加载器。

        """
        # Set mode to 0 and extract data
        ra_src = self.RA_path.split('/')[-3]
        s = "[" + "-".join(map(str, self.ratio)) + "]"
        unique_dataset_name = ra_src + "_" + s
        unique_topkdata_name = self.model_dir + "_" + unique_dataset_name
        path = os.path.join(self.save_dir, self.topk_dir, unique_topkdata_name)
        if os.path.exists(path) is False:
            os.mkdir(path)
        filename = os.path.join(path, f"{mode}.csv")
        
        dataset.setmode(0)
        dataset.extract_data()

        if os.path.isfile(filename):
            _df = pd.read_csv(filename)
            _df['data'] = _df['data'].apply(eval)
            dataset.df = _df
            dataset.setmode(1)
            dataset.extract_data()
        else:
            # Create a DataLoader for the dataset
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            # Perform inference to get top-k list
            topk_list = inference(model, dataloader, self.device, topk)
            # Set mode to 1, extract top-k data
        
            dataset.setmode(1)
            dataset.extract_data(k=topk, topk=topk_list)
            # Save the extracted data to CSV
            dataset.df_extract.to_csv(filename, index=False)
        
        # Create a new DataLoader for the modified dataset
        shuffle = True if mode == 'train' else False
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

        
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        return self.loss(outputs, labels)

def read_files(fileslist):
    # 初始化一个空的DataFrame列表
    dfs = []
    # 读取每个文件并添加到列表中
    for file in fileslist:
        df = pd.read_csv(file)
        dfs.append(df)

    # 合并所有的DataFrame
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    return result




def find_topk(res, topk):
    # 按照v1值对列表进行排序，为分组做准备
    res_sorted = sorted(res, key=lambda x: x[5])
    
    # 结果列表
    result = []
    print("start find topk")
    # 按v1值进行分组
    for v1, group in groupby(res_sorted, key=lambda x: x[5]):
        # 在每个分组内按照v2值进行排序（降序），提取前topk个元素
        topk_tuples = sorted(group, key=lambda x: float(x[6]), reverse=True)[:topk]
        # if len(topk_tuples) != topk:
        #     print(f"v1: {v1} has {len(topk_tuples)} elements")
        # 将提取的元组添加到结果列表中，但是只包括key和v1
        if len(topk_tuples) != topk:
            continue
        result.extend([t for t in topk_tuples])
    
    return result


def inference(model, dataloader, device, topk, mode="4mer"):
    model.eval()
    res = []
    with torch.no_grad():
        for batch_idx, (data, _4mer_RA, tcr_RA, names, labels, samples) in tqdm(enumerate(dataloader), desc=\
                                                                                "Inferencing", total=len(dataloader)):
            # Rest of the code
            data = data.to(device)
            if mode == "4mer":
                RA = _4mer_RA.view(-1, 1).to(device)
            else:
                RA = tcr_RA.view(-1, 1).to(device)
            outputs = model(data, RA)
            outputs = F.softmax(outputs, dim=1)
            outputs = outputs.cpu()
            # 将 outputs 转换为一维并转换为 Python 列表
            # outputs_list = outputs.view(-1).tolist()
            b_l = labels.size(0)
            labels_list = labels.view(-1).tolist()
            # 向列表中插入新的键值对
            # for key, v1, v2, labels in zip(names, samples, outputs_list, labels_list):
            #     res.append((key, v1, v2, labels))
            for j in range(b_l):
                res.append((data[j].cpu().view(4,5).tolist(), _4mer_RA[j].cpu().item(), tcr_RA[j].cpu().item(),\
                    names[j], labels_list[j], samples[j], outputs[j][labels_list[j]].item()))
        topk_res = find_topk(res, topk)

    return topk_res




def train_MIL(model, dataloader, criterion, device, lr_decay, optimizer, scaler, lr_scheduler, mode="4mer"):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for batch_idx, (data, _4mer_RA, _tcr_RA, names, labels, samples) in tqdm(enumerate(dataloader), desc=\
                                                                                "Training", total=len(dataloader)):
        lr_decay.append(optimizer.state_dict()["param_groups"][0]["lr"])
        data = data.to(device)
        if mode == "4mer":
            RA = _4mer_RA.view(-1, 1).to(device)
        else:
            RA = _tcr_RA.view(-1, 1).to(device)
        labels = labels.view(-1,).to(device)
        # labels = torch.full((data.size(0), 1), label, device=device, dtype=torch.float32)
        optimizer.zero_grad()
        with autocast():
            outputs = model(data, RA)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        b_l = labels.size(0)
        total += b_l
        outputs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        correct = torch.eq(predicted, labels).sum().item()
        running_correct += correct
        print(f"Batch {batch_idx + 1}/{len(dataloader)} Loss: {loss.item() / b_l:.6f} Acc: {correct / b_l:.6f}", end="\r")
        lr_scheduler.step()
    return running_loss / total, running_correct / total

def train_RNN(epoch, embedder, rnn, dataloader, criterion, optimizer, device, mode="4mer"):
    rnn.train()
    running_loss = 0.
    running_correct = 0
    correct = 0
    total = 0

    for batch_idx,(data, _4mer_RA, _tcr_RA, names, labels, samples) in enumerate(dataloader):
        print('Training - Epoch: [{}] Batch: [{}/{}]'.format(epoch+1, batch_idx+1, len(dataloader)))
        print(" ")
        data = data.to(device)
        
        if mode == "4mer":
            RA = _4mer_RA.view(-1,  data.shape[1], 1).to(device)
        else:
            RA = _tcr_RA.view(-1, data.shape[1], 1).to(device)
        labels = labels[:,0].to(device)
        b_l = labels.size(0)
        stacked = []
        optimizer.zero_grad()
        for s in range(data.shape[1]):
            #inp (B, hidden_size)
            inp = embedder(data[:, s, :], RA[:, s, :])
            stacked.append(inp)
        
        # inps (B, S, hidden_size)
        inps = torch.stack(stacked, dim=1)
        output = rnn(inps)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        running_loss += loss
        total += b_l
        output = F.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        correct = torch.eq(predicted, labels).sum().item()
        running_correct += correct
        print("Batch",batch_idx + 1/len(dataloader), "Loss:", loss / b_l, "Acc:",  correct / b_l)
    return running_loss / total, running_correct / total

def test_RNN(epoch, embedder, rnn, dataloader, criterion, device, save_dir = '/xiongjun/test/MIL/figs',mode="4mer", picname = "ROC"):
    rnn.eval()
    running_loss = 0.
    running_correct = 0
    correct = 0
    total = 0
    accu_out = []
    accu_labels = []
    with torch.no_grad():
        for batch_idx,(data, _4mer_RA, _tcr_RA, names, labels, samples) in enumerate(dataloader):
            print('validating - Epoch: [{}] Batch: [{}/{}]'.format(epoch+1, batch_idx+1, len(dataloader)))
            data = data.to(device)
            
            if mode == "4mer":
                RA = _4mer_RA.view(-1, data.shape[1], 1).to(device)
            else:
                RA = _tcr_RA.view(-1, data.shape[1], 1).to(device)
            labels = labels[:,0].to(device)
            stacked = []
            for s in range(data.shape[1]):
                #inp (B, hidden_size)
                inp = embedder(data[:, s, :], RA[:, s, :])
                stacked.append(inp)
            
            # inps (B, S, hidden_size)
            inps = torch.stack(stacked, dim=1)
            
            # output (B, 2)
            output = rnn(inps)
            loss = criterion(output, labels)
            loss = loss.item()
            b_l = labels.size(0)
            total += b_l
            output = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            correct = torch.eq(predicted, labels).sum().item()
            running_loss += loss
            running_correct += correct
            print("Batch",batch_idx + 1/len(dataloader), "Loss:", loss / b_l, "Acc:",  correct / b_l)
            labels = labels.cpu()
            # 对每个类别绘制ROC曲线
            output = output.cpu()
            accu_out.append(output)
            accu_labels.append(labels)

        # 使用torch.cat将它们堆叠起来
        total_out = torch.cat(accu_out, dim=0).numpy()
        total_labels = torch.cat(accu_labels, dim=0).numpy()
        for i in range(7):
            fpr, tpr, _ = roc_curve(total_labels == i, total_out[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        folder = os.path.join(save_dir, 'figs')
        if os.path.exists(folder) is False:
            os.mkdir(folder)
        plt.savefig(os.path.join(folder, picname + '.png'))

    return running_loss / total, running_correct / total