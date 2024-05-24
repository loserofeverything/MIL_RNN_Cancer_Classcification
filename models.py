import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import matplotlib.pyplot as plt
import numpy as np

class FeatureExtractor(nn.Module):
    """
    Neural network model for binary classification.
    """

    def __init__(self, input_dim, output_dim):
        """
        Initialize the model.

        Args:
            input_size (int): The input size of the model.
        """
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim // 2)
        self.fc2 = nn.Linear(1, output_dim // 2)

    def forward(self, x, RA):
        """
        Forward pass of the model.

        Args:
            x (tensor): The input tensor.
            RA (tensor): The RA value.

        Returns:
            tensor: The output tensor.
        """
        x = self.fc1(x)
        ra = self.fc2(RA)
        return torch.cat((x, ra), dim=1)


class FeatureExtractor2(nn.Module):
    def __init__(self, hidden_dim):
        super(FeatureExtractor2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)  # 第一个卷积层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1) # 第二个卷积层
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0) # 池化层
        self.fc1 = nn.Linear(128, hidden_dim // 2)         # 隐藏层
        self.fc2 = nn.Linear(1, hidden_dim//2)

    def forward(self, x, RA):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积层1 + 激活函数 + 池化
        ra = self.fc2(RA)
        x = self.pool(F.relu(self.conv2(x)))  # 卷积层2 + 激活函数 + 池化
        x = x.view(-1, 128)            # 展平
        x = self.fc1(x)               
        x = torch.cat((x, ra), dim=1) # 全连接层1 + 激活函数
        return x
    
class FeatureExtractor3(nn.Module):
    def __init__(self, hidden_dim):
        super(FeatureExtractor3, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 8, hidden_dim // 2)  # 假设池化后长度为8
        self.fc2 = nn.Linear(1, hidden_dim // 2)
        

    def forward(self, x, RA):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 16*8)
        x = self.fc1(x)
        ra = self.fc2(RA)
        x = torch.cat((x, ra), dim=1)
        return x

class convTCRNet(nn.Module):
    def __init__(self, hidden_dim, num_class):
        super(convTCRNet, self).__init__()
        self.emb = FeatureExtractor2(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_class)          # 输出层 (假设分类任务有10类)

    def forward(self, x, RA):
        x = self.emb(x, RA)
        x = F.relu(x) # 全连接层1 + 激活函数
        x = self.fc(x)                       # 输出层
        return x

class conv1dTCRNet(nn.Module):
    def __init__(self, hidden_dim, num_class):
        super(conv1dTCRNet, self).__init__()
        self.emb = FeatureExtractor3(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_class)          # 输出层 (假设分类任务有10类)

    def forward(self, x, RA):
        x = self.emb(x, RA)
        x = F.relu(x) # 全连接层1 + 激活函数
        x = self.fc(x)                       # 输出层
        return x


class MODEL(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MODEL, self).__init__()
        self.emb = FeatureExtractor(input_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, RA):
        return self.fc(self.emb(x, RA))

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device = 'cpu'):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, num_classes).to(device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
class GAttNet(nn.Module):
    def __init__(self, input_dim, instance_hidden_dim, instance_concepts, bag_hidden_dim, num_class):
        super(GAttNet, self).__init__()
        self.W = nn.Parameter(torch.Tensor(instance_hidden_dim, instance_concepts))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.embs = FeatureExtractor(input_dim, instance_hidden_dim)
        self.V = nn.Parameter(torch.Tensor(instance_hidden_dim, bag_hidden_dim))
        init.kaiming_uniform_(self.V, a=math.sqrt(5))
        self.wB = nn.Parameter(torch.Tensor(bag_hidden_dim, 1))
        init.kaiming_uniform_(self.wB, a=math.sqrt(5))
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(bag_hidden_dim, num_class)
            )

    def forward(self, x, ra):
        fx = self.embs(x, ra)
        HI = torch.mm(fx.t(), F.softmax(torch.mm(fx, self.W), dim =1))
        HB = torch.mm(self.V.t(), HI)
        b = torch.mm(HB, F.softmax(torch.mm(HB.t(), self.wB), dim = 1)).t()
        outputs = self.fc(b)
        return outputs, HI


# random_tensor = torch.randn(100, 20)
# random_ra = torch.randn(100, 1)
# gattnet = GAttNet(20, 128, 32, 64,7)
# output = gattnet(random_tensor, random_ra)
# print(output.size())


class Inst2BagPooling(nn.Module):
    def __init__(self, instance_embedding_dim, num_instance_concepts, bag_embedding_dim):
        super(Inst2BagPooling, self).__init__()
        self.num_instance_concepts = num_instance_concepts
        self.instance_concept_weights = nn.Parameter(torch.Tensor(instance_embedding_dim, num_instance_concepts))
        self.bag_transform_weights = nn.Parameter(torch.Tensor(instance_embedding_dim, bag_embedding_dim))
        self.bag_concept_weights = nn.Parameter(torch.Tensor(bag_embedding_dim, 1))
        self._initialize_weights()
    
    def _initialize_weights(self):
        init.xavier_uniform_(self.instance_concept_weights)
        init.xavier_uniform_(self.bag_transform_weights)
        init.xavier_uniform_(self.bag_concept_weights)

    def forward(self, f_X):
        B = f_X.size(0)
        num_instances = f_X.size(1)

        # Step 1: Initialize instance-level target concepts
        W_I = self.instance_concept_weights
        
        # Step 2: Generate a new bag from pooling on f_X with instance-level target concepts
        attention_weights_I = torch.softmax(torch.matmul(f_X, W_I), dim=2)  # shape: (B, 20, num_instance_concepts)
        H_I = torch.matmul(f_X.transpose(1, 2), attention_weights_I).transpose(1, 2)  # shape: (B, 20, num_instance_concepts)
        
        # Step 3: Map the new bag from the low-level instance space into the high-level bag space
        V = self.bag_transform_weights
        H_B = torch.matmul(H_I, V)  # shape: (B, 20, bag_embedding_dim)
        
        # Step 4: Initialize a bag-level concept
        W_B = self.bag_concept_weights
        
        # Step 5: Generate the final bag vector from pooling on H_B with the bag-level concept
        attention_weights_B = torch.softmax(torch.matmul(H_B, W_B), dim=1)  # shape: (B, inst_conp, 1)
        b = torch.matmul(H_B.transpose(1, 2), attention_weights_B).squeeze(-1)  # shape: (B, bag_embedding_dim)
        attn = torch.matmul(attention_weights_I, attention_weights_B)
        return b, H_I, attn

class BGAttNet(nn.Module):
    def __init__(self, input_dim, instance_embedding_dim, num_instance_concepts, bag_embedding_dim, num_class):
        super(BGAttNet, self).__init__()
        self.inst2bag_pooling = Inst2BagPooling(instance_embedding_dim, num_instance_concepts, bag_embedding_dim)
        self.embs = FeatureExtractor(input_dim, instance_embedding_dim)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(bag_embedding_dim, num_class)
        )

    def forward(self, x, ra):
        B, k, input_dim = x.size()
        x = x.view(B * k, input_dim)
        ra = ra.view(B * k, -1)
        fx = self.embs(x, ra)
        fx = fx.view(B, k, -1)
        b, H_I, attn = self.inst2bag_pooling(fx)
        outputs = self.fc(b)
        return outputs, H_I, attn

# # Example usage
# instance_embedding_dim = 128
# num_instance_concepts = 10
# bag_embedding_dim = 64
# input_dim = 20

# bgattnet = BGAttNet(input_dim, instance_embedding_dim, num_instance_concepts, bag_embedding_dim, 7)
# B = 2  # example batch size
# x = torch.randn(B, 20, 20)
# ra = torch.randn(B, 20, 1)
# outpus, HI, attn = bgattnet(x, ra)
# influential_instances = highlight_influential_instances(x, attn, top_k=3)

# # print(influential_instances)

# visualize_attention_weights(attn.squeeze(-1), title="Attention Weights")



class TCRConv1DModel(nn.Module):
    def __init__(self):
        super(TCRConv1DModel, self).__init__()
        # 定义1D卷积层
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 8, 128)  # 假设池化后长度为8
        self.fc2 = nn.Linear(128, 64)  # 最后的特征向量维度

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 8)  # 展平成全连接层输入
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x




# 定义一个简单的实例分类模型
class InstanceClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(InstanceClassifier, self).__init__()
        # 定义1D卷积层
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 8, hidden_dim // 2)  # 假设池化后长度为8
        self.fc2 = nn.Linear(hidden_dim, num_classes)  # 最后的特征向量维度
        self.fc3 = nn.Linear(1, hidden_dim // 2)

    def forward(self, x, RA):
        b = x.size(0)  # batch size
        x = x.permute(0, 1, 3, 2) 
        x = x.view(-1, 5, 16)  # 展平成1D卷积层输入
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 8)  # 展平成全连接层输入
        x = self.fc1(x)
        ra = self.fc3(RA)
        x = torch.cat((x, ra), dim=1)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.view(b,-1,self.num_classes)
        return x

# 定义一个带有注意力机制的多示例学习模型
class AttentionMIL(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super(AttentionMIL, self).__init__()
        self.instance_classifier = InstanceClassifier(hidden_dim, num_classes)
        self.attention = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, RA):
        #x: (B, instance_counts, seq_length, aa_dim)
        #RA: (B, instance_counts, 1)
        B, k, _,_ = x.size()
        RA = RA.view(B * k, -1)
        instance_preds = self.instance_classifier(x, RA)
        attention_scores = self.attention(instance_preds)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_instance_preds = instance_preds * attention_weights
        bag_logits = weighted_instance_preds.sum(dim=1)
        #bag_logits: (B, num_classes)
        #attention_weights: (B, instance_counts, 1)
        return bag_logits, attention_weights



# 定义一个带有注意力机制的多示例学习模型
class SelfAttentionMIL(nn.Module):

    def __init__(self, hidden_dim, num_classes, num_heads=7):
        super(SelfAttentionMIL, self).__init__()
        self.instance_classifier = InstanceClassifier(hidden_dim, num_classes)
        self.self_attention = nn.MultiheadAttention(embed_dim=num_classes, num_heads=num_heads)
        self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, x, RA):
        # x: (B, instance_counts, seq_length, aa_dim)
        # RA: (B, instance_counts, 1)
        B, k, _, _ = x.size()
        RA = RA.view(B * k, -1)
        instance_preds = self.instance_classifier(x, RA)

        # 将instance_preds转换为 (instance_counts, B, num_classes) 以适应MultiheadAttention的输入
        instance_preds = instance_preds.permute(1, 0, 2)
        
        # 自注意力机制
        attention_output, attention_weights = self.self_attention(instance_preds, instance_preds, instance_preds)
        
        # 将注意力输出转换回 (B, instance_counts, num_classes)
        attention_output = attention_output.permute(1, 0, 2)

        # 聚合实例的预测结果
        weighted_instance_preds = self.fc(attention_output)
        bag_logits = weighted_instance_preds.mean(dim=1)

        # bag_logits: (B, num_classes)
        # attention_weights: (instance_counts, B, instance_counts)
        return bag_logits, attention_weights

    def explain_instance_importance(self, attention_weights):
        # 计算每个实例的重要性
        # 对于每个包中的实例，取所有注意力头的平均值
        instance_importance = attention_weights.mean(dim=1)
        return instance_importance




# 局部注意力机制
class LocalAttention(nn.Module):
    def __init__(self, num_classes, local_radius=5):
        super(LocalAttention, self).__init__()
        self.local_radius = local_radius
        self.attention = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        B, k, C = x.size()
        attention_scores = torch.zeros(B, k, k).to(x.device)
        
        for i in range(k):
            local_indices = range(max(0, i - self.local_radius), min(k, i + self.local_radius + 1))
            local_features = x[:, local_indices, :]
            local_scores = self.attention(local_features)  # 计算局部注意力得分
            attention_scores[:, i, local_indices] = local_scores.squeeze(-1)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_instance_preds = torch.bmm(attention_weights, x)  # (B, k, k) x (B, k, C) -> (B, k, C)
        return weighted_instance_preds, attention_weights

# 定义一个带有局部注意力机制的多示例学习模型
class LocalAttentionMIL(nn.Module):
    def __init__(self, hidden_dim, num_classes, local_radius=5):
        super(LocalAttentionMIL, self).__init__()
        self.instance_classifier = InstanceClassifier(hidden_dim, num_classes)
        self.local_attention = LocalAttention(num_classes, local_radius)
        self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, x, RA):
        # x: (B, instance_counts, seq_length, aa_dim)
        # RA: (B, instance_counts, 1)
        B, k, _, _ = x.size()
        RA = RA.view(B * k, -1)
        instance_preds = self.instance_classifier(x, RA)
        
        weighted_instance_preds, attention_weights = self.local_attention(instance_preds)
        
        bag_logits = weighted_instance_preds.mean(dim=1)

        # bag_logits: (B, num_classes)
        # attention_weights: (B, instance_counts, instance_counts)
        return bag_logits, attention_weights

    def explain_instance_importance(self, attention_weights):
        instance_importance = attention_weights.mean(dim=1)
        return instance_importance







# 可视化函数
def plot_instance_importance(instance_importance, bag_index):
    """
    可视化实例重要性评分
    :param instance_importance: 实例重要性评分张量, 形状为 (B, instance_counts)
    :param bag_index: 需要可视化的包的索引
    """
    instance_scores = instance_importance[bag_index].cpu().detach().numpy()
    instance_indices = np.arange(len(instance_scores))
    
    plt.figure(figsize=(10, 6))
    plt.bar(instance_indices, instance_scores, color='blue')
    plt.xlabel('Instance Index')
    plt.ylabel('Importance Score')
    plt.title(f'Instance Importance Scores for Bag {bag_index}')
    plt.savefig('/xiongjun/test/MIL/instance_importance.png')


# # 使用示例
# hidden_dim = 128
# num_classes = 7
# model = LocalAttentionMIL(hidden_dim, num_classes)
# x = torch.randn(4, 1000, 16, 5)  # (B, instance_counts, seq_length, aa_dim)
# RA = torch.randn(4, 1000, 1)  # (B, instance_counts, 1)
# bag_logits, attention_weights = model(x, RA)
# instance_importance = model.explain_instance_importance(attention_weights)
# print(bag_logits.shape)  # 输出: torch.Size([4, 10])
# print(attention_weights.shape)  # 输出: torch.Size([10, 4, 10])
# print(instance_importance.shape)  # 输出: torch.Size([4, 10])

# # 可视化第一个包中的实例重要性评分
# plot_instance_importance(instance_importance, bag_index=0)