import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import re
import matplotlib.pyplot as plt
import random
import torch
import torch.optim as optim
from utils import  MyLoss, Preparation, test_RNN
from models import RNN
from utils import rnndataset, read_files, inference
import pickle
from torch.utils.data import DataLoader

# 定义数据
data = "A,-0.591,-1.302,-0.733,1.570,-0.146;C,-1.343,0.465,-0.862,-1.020,-0.255;D,1.050,0.302,-3.656,-0.259,-3.242;E,1.357,-1.453,1.477,0.113,-0.837;F,-1.006,-0.590,1.891,-0.397,0.412;G,-0.384,1.652,1.330,1.045,2.064;H,0.336,-0.417,-1.673,-1.474,-0.078;I,-1.239,-0.547,2.131,0.393,0.816;K,1.831,-0.561,0.533,-0.277,1.648;L,-1.019,-0.987,-1.505,1.266,-0.912;M,-0.663,-1.524,2.219,-1.005,1.212;N,0.945,0.828,1.299,-0.169,0.933;P,0.189,2.081,-1.628,0.421,-1.392;Q,0.931,-0.179,-3.005,-0.503,-1.853;R,1.538,-0.055,1.502,0.440,2.897;S,-0.228,1.399,-4.760,0.670,-2.647;T,-0.032,0.326,2.213,0.908,1.313;V,-1.337,-0.279,-0.544,1.242,-1.262;W,-0.595,0.009,0.672,-2.128,-0.184;Y,0.260,0.830,3.097,-0.838,1.512"

# 将数据分割成列表
data_list = [item.split(',') for item in data.split(';')]

# 创建DataFrame
atchley = pd.DataFrame(data_list, columns=["amino.acid", "f1", "f2", "f3", "f4", "f5"])

# 将f1-f5列转换为数值类型
atchley[["f1", "f2", "f3", "f4", "f5"]] = atchley[["f1", "f2", "f3", "f4", "f5"]].apply(pd.to_numeric)

# 将amino.acid列设置为索引
atchley.set_index("amino.acid", inplace=True)
atchley_dict = {row[0]: list(row[1:]) for row in atchley.itertuples()}

# 定义一个函数，将氨基酸序列转换为数值列表
def sequence_to_values(sequence):
    return [atchley_dict[amino_acid] for amino_acid in sequence]

def calRA(raw_data, sample_name, keep, types, RA_save_dir ,amino_acids=set('ACDEFGHIKLMNPQRSTVWY'),\
          values = ['breast', 'prostate', 'lung', 'liver', 'pancreas', 'colorectal', 'health']):
    if len(raw_data) <= 1000:
        return
    raw_data['aaSeqCDR3_length'] = raw_data['aaSeqCDR3'].str.len()

    # 计算所有aaSeqCDR3长度相同的行中的cloneCount的和
    grouped = raw_data.groupby('aaSeqCDR3_length')['cloneFraction'].sum()

    # 对grouped进行排序
    sorted_grouped = grouped.sort_values(ascending=False)

    # 计算累积和
    cumsum = sorted_grouped.cumsum()

    # 找到保持原来cloneCount总数100*keep%以上的那些行
    mask = cumsum <= cumsum.iloc[-1] * keep
    if not mask.any():  # 如果 mask 全为 False
        mask.iloc[0] = True  # 将第一行的值设为 True
    filtered_grouped = sorted_grouped[mask]
    data_filtered = raw_data[raw_data['aaSeqCDR3_length'].isin(filtered_grouped.index)]
    num_rows_filtered = len(data_filtered)
    max_length = data_filtered['aaSeqCDR3_length'].max()
    del raw_data
    # 去掉第一个和最后三个值
    data_filtered['aaSeqCDR3'] = data_filtered['aaSeqCDR3'].apply(lambda x: x[1:-3])
    # 找到aaSeqCDR3中元素的最大长度
    max_length = data_filtered['aaSeqCDR3'].str.len().max()
    # 使用'-'在末尾进行填充到aaSeqCDR3中元素的最大长度
    data_filtered['aaSeqCDR3'] = data_filtered['aaSeqCDR3'].apply(lambda x: x.ljust(max_length, '-'))
    length = len(data_filtered['aaSeqCDR3'].iloc[0])

    # 对于每个4-mer序列
    for i in range(length - 3):
        # 创建新的列
        data_filtered[f'4-mer-{i+1}'] = data_filtered['aaSeqCDR3'].apply(lambda x: x[i:i+4] if set(x[i:i+4]).issubset(amino_acids) else np.nan)

    
    df_backup = data_filtered.copy()

    # 找到所有的 '4-mer' 列
    four_mer_columns = data_filtered.filter(regex='4-mer')

    # 将 '4-mer' 列和 'ratio' 列合并
    df_melted = pd.melt(data_filtered, id_vars='cloneFraction', value_vars=four_mer_columns.columns, var_name='4-mer_col', value_name='4-mer')
    # 删除4-mer列中的NaN值
    df_melted = df_melted.dropna(subset=['4-mer'])
    # 计算每种4-mer的最大ratio值
    TCR_RA_stats = df_melted.groupby('4-mer')['cloneFraction'].max().reset_index()
    TCR_RA_stats.columns = ['4-mer', 'RA']


    data_filtered = df_backup
    # 初始化一个字典来存储每种4-mer序列的相对丰度值
    relative_abundance = {}

    # 遍历每一行数据
    for index, row in data_filtered.iterrows():
        # 遍历除了最后一列（cloneFraction）以外的所有列
        for col in four_mer_columns.columns:
            # 如果该列的值不是NaN，则将其相对丰度值累加到相应的键中
            if not pd.isna(row[col]):
                # 使用setdefault方法来初始化字典中键的默认值为0，然后累加cloneFraction值
                relative_abundance.setdefault(row[col], 0)
                relative_abundance[row[col]] += row['cloneFraction']



    _4mer_RA_stats = pd.DataFrame(list(relative_abundance.items()), columns=['4-mer', 'RA'])
    sumra = _4mer_RA_stats['RA'].sum()
    _4mer_RA_stats['RA'] = _4mer_RA_stats['RA'] / sumra
    RA_stats = _4mer_RA_stats.merge(TCR_RA_stats, on='4-mer', suffixes=('_4mer', '_TCR'))

    # 使用 '4-mer' 列的值创建新的列
    RA_stats['data'] = RA_stats['4-mer'].apply(sequence_to_values)
    RA_stats['sample'] = sample_name
    label = values.index(types)
    RA_stats['label'] = label
    RA_stats.set_index('4-mer', inplace=True)
    return RA_stats

def readRA(filename, keep, cancer_type):
    # 读取文件
    path_parts = os.path.split(filename)
    name = path_parts[1]
    raw_data = pd.read_csv(filename, sep='\t')
    raw_data['sample'] = name.split('.')[0] 
    types = cancer_type
    # 创建一个新的列来存储aaSeqCDR3列中元素的长度
    df_group = raw_data.groupby('sample')
    RA_save = str(keep) + '_RA'
    for  sample_id, (sample_name, df) in enumerate(df_group):
        RA = calRA(df, sample_name, keep, types, RA_save)
    return RA

class siglefiletest(Preparation):
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
        with open(os.path.join(path, 'train.pkl'), 'wb') as f:
            pickle.dump(train_dataset, f)
        return train_dataset
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
        
        # Create a new DataLoader for the modified dataset
        shuffle = True if mode == 'train' else False
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader


cancer_type = 'lung'
missingfile = "H2001H023.clonotypes.TRB.txt"
directory = "/xiongjun/test/MIL/share/MixResult_UID_All"
file_path = os.path.join(directory, missingfile)
RA = readRA(file_path, 0.9, cancer_type)
values = ['breast', 'prostate', 'lung', 'liver', 'pancreas', 'colorectal', 'health']
tid = values.index(cancer_type)

if os.path.exists("/xiongjun/test/MIL/testsinglefile/{}".format(tid)) is False:
    os.mkdir("/xiongjun/test/MIL/testsinglefile/{}".format(tid))

RA.to_csv("/xiongjun/test/MIL/testsinglefile/{}/RA.csv".format(tid))



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

mode = "4mer"
epochs = 10
batch_size = 128
batch_size2 = 1
# rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=2, batch_first=True).to(device)
rnn = RNN(128, 256, 2, 7, device)
criterion = MyLoss()
optimizer = optim.SGD(rnn.parameters(), 0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
RA_path = "/xiongjun/test/MIL/testsinglefile/*/*.csv"
MIL_dir = "40_128_4mer"
test_save_dir = "/xiongjun/test/MIL/testsinglefile"

prepare = siglefiletest(test_save_dir, MIL_dir, RA_path,\
        device = device,ratio = [1.0, 0.0, 0.0])

# load or make dataset
train_dataset = prepare.loadData()

model = prepare.loadModel(20, 128, 7)
embs = model.emb
train_dataloader = prepare.extractDataLoader(train_dataset, model, 1, 40, "train")
rnn.load_state_dict(torch.load("/xiongjun/test/MIL/testsinglefile/RNN/40_128_4mer_256_2/RNN-0-1.0.pth"))

_loss_, _acc_ = test_RNN(114514, embs, rnn, train_dataloader, criterion, device, choice = "False")
print(_loss_, _acc_)

