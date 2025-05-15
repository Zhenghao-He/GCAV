
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tcav.tcav import TCAV
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import random
from align_dim import augment_cavs
from configs import  fuse_input, mymodel, source_dir, activation_dir, k1, k2, nce_loss, con_loss, var_loss, sim_loss
# from configs import num_random_exp

def normalize_tensor(x, eps=1e-8):
    """
    标准化单个张量。
    
    :param x: 输入张量，形状为 [embedding_dim]
    :param eps: 小常数，避免除零
    :return: 标准化后的张量
    """
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)

def normalize_cav(cavs, eps=1e-8):
    """
    标准化CAV张量。
    
    :param cavs: 输入CAV张量形状为 [num_layers, num_cavs, embedding_dim] 或 [num_cavs, embedding_dim]
    :param eps: 小常数，避免除零
    :return: 标准化后的CAV张量
    """
    result = []
    for layer_cavs in cavs:
        tmp = []
        for cav in layer_cavs:
            cav1 = normalize_tensor(cav, eps)
            tmp.append(cav1)
        result.append(tmp)
    return result



  



def cosine_similarity(a, b):
    dot_product = np.dot(a, b)  
    norm_a = np.linalg.norm(a)  
    norm_b = np.linalg.norm(b)  
    return dot_product / (norm_a * norm_b)

def prepare_batch(cavs,device,num_random_exp=3,batch_size=2):    
    """
    
    Args:
        cavs:  [n_layers, n_concepts, cav_dim]
        concept_idx: 
    Returns:
        cav_query: Tensor
        cav_key: Tensor
    """
    
    # concept_idx = random.randint(0, len(cavs[0])//num_random_exp-1)  #random sample one concept
    # concept_indices = [concept_idx*num_random_exp + i for i in range(num_random_exp)]
    # concepts_tensors = [cav[i] for cav in cavs for i in concept_indices]
    # # concepts_tensors = [cav[concept_indices] for cav in cavs]
    # # import pdb; pdb.set_trace()
    # concepts_arr = [cav.cpu().numpy() for cav in concepts_tensors]

    # half_len = len(concepts_arr) // 2
    # if batch_size > half_len:
    #     raise ValueError("Batch size is larger than the number of concepts.")
    # cav_query = random.sample(concepts_arr, batch_size)
    # # import pdb; pdb.set_trace()
    # cav_key = [x for x in concepts_arr if not any(np.array_equal(x, y) for y in cav_query)]
    # if len(cav_key) > batch_size:
    #     cav_key = random.sample(cav_key, batch_size)

    # aug_query = augment_cavs(cavs=cav_query, num_augments=2, noise_std=0.1) 
    # aug_query = np.concatenate((aug_query, cav_query), axis=0)
    # aug_query = [torch.tensor(cav, dtype=torch.float32, device=device) for cav in aug_query]
    # aug_key = augment_cavs(cavs=cav_key, num_augments=2, noise_std=0.1) 
    # aug_key = np.concatenate((aug_key, cav_key), axis=0)
    # aug_key = [torch.tensor(cav, dtype=torch.float32, device=device) for cav in aug_key]


    n_layers = len(cavs)
    layer_query, layer_key = random.sample(range(n_layers), 2)  #random sample two layers
    cav_q_arr = np.array([cav.cpu().numpy() for cav in cavs[layer_query]])
    cav_k_arr = np.array([cav.cpu().numpy() for cav in cavs[layer_key]])
    aug_query = augment_cavs(cavs=cav_q_arr, num_augments=1, noise_std=0.1)
    aug_key = augment_cavs(cavs=cav_k_arr, num_augments=1, noise_std=0.1)

    aug_query = [torch.tensor(cav, dtype=torch.float32, device=device) for cav in aug_query]
    aug_key = [torch.tensor(cav, dtype=torch.float32, device=device) for cav in aug_key]

    cav_query = torch.stack(aug_query) # [n_concept_tensors, tensor_dim]
    cav_key = torch.stack(aug_key)  
    return cav_query, cav_key

# class InfoNCELoss(nn.Module):
#     def __init__(self, temperature=0.1):
#         super().__init__()
#         self.temperature = temperature  # 温度参数
    
#     def forward(self, anchor, positive, negatives):
#         """
#         Args:
#             anchor: [batch_size, embed_dim] 锚点样本
#             positive: [batch_size, embed_dim] 正样本
#             negatives: [batch_size, num_neg, embed_dim] 负样本
#         Returns:
#             loss: InfoNCE 损失值
#         """
#         # 计算锚点与正样本的相似度
#         pos_sim = F.cosine_similarity(anchor, positive, dim=1)  # [batch_size]
#         pos_sim = pos_sim / self.temperature
        
#         # 计算锚点与负样本的相似度
#         anchor_expanded = anchor.unsqueeze(1)  # [batch_size, 1, embed_dim]
#         neg_sim = F.cosine_similarity(anchor_expanded, negatives, dim=2)  # [batch_size, num_neg]
#         neg_sim = neg_sim / self.temperature
        
#         # 计算 InfoNCE 损失
#         numerator = torch.exp(pos_sim)  # [batch_size]
#         denominator = numerator + torch.exp(neg_sim).sum(dim=1)  # [batch_size]
#         loss = -torch.log(numerator / denominator).mean()
#         return loss
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5, reg_lambda=0):
        super().__init__()
        self.temperature = temperature
        self.reg_lambda = reg_lambda  # 正则化项权重
    
    def forward(self, anchor, positive, negatives):
        # 原始 InfoNCE 计算
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        pos_sim_scaled = pos_sim / self.temperature
        
        anchor_expanded = anchor.unsqueeze(1)
        neg_sim = F.cosine_similarity(anchor_expanded, negatives, dim=2)
        neg_sim_scaled = neg_sim / self.temperature
        
        numerator = torch.exp(pos_sim_scaled)
        denominator = numerator + torch.exp(neg_sim_scaled).sum(dim=1)
        nce_loss = -torch.log(numerator / denominator).mean()
        
        # 添加正则化项：强制正样本相似度接近1
        reg_term = (1 - pos_sim).mean()  # 最小化 1 - 余弦相似度
        total_loss = nce_loss + self.reg_lambda * reg_term
        
        return total_loss
    
class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, anchor, positive, ori_anchor, ori_positive):
        pos_ori_sim1 = F.cosine_similarity(anchor, ori_anchor, dim=1)
        pos_ori_sim2 = F.cosine_similarity(positive, ori_positive, dim=1)
        pos_loss = (1-pos_ori_sim1).mean() + (1-pos_ori_sim2).mean()
        return pos_loss


class TransformerQueryEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers=2, num_heads=4, ff_dim=2048, dropout=0.1):
        """
        Transformer-based query encoder for MoCo.
        Args:
            embed_dim: Dimension of the input and output embeddings.
            num_layers: Number of Transformer encoder layers.
            num_heads: Number of attention heads in the Transformer.
            ff_dim: Dimension of the feed-forward network.
            dropout: Dropout rate for regularization.
        """
        super(TransformerQueryEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="relu",
                batch_first=True
            ),
            num_layers=num_layers
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, embed_dim].
        Returns:
            Transformed query embeddings of shape [batch_size, embed_dim].
        """
        # Apply Transformer
        x = self.transformer(x)
        return x[:, 0, :]  # Return the first token embedding as the query
    

class CAVDataset(Dataset):
    """
    自定义数据集，用于存储 CAV 嵌入和标签。
    """
    def __init__(self, cav_embeddings, labels):
        """
        Args:
            cav_embeddings: Tensor of shape [num_samples, num_layers, embedding_dim].
            labels: Tensor of shape [num_samples].
        """
        super(CAVDataset, self).__init__()
        self.cav_embeddings = cav_embeddings  # 输入数据
        self.labels = labels  # 对应的标签

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cav_embeddings[idx], self.labels[idx]
    

class OrthogonalLinear(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(OrthogonalLinear, self).__init__()
        # self.fc = nn.Linear(input_dim, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return F.normalize(x, dim=1)

    def orthogonal_regularization(self, weight_decay=1e-1):
        """Apply orthogonal regularization to the weight matrix."""
        W = self.fc.weight
        loss = torch.norm(torch.mm(W, W.T) - torch.eye(W.size(0), device=W.device))
        return weight_decay * loss
'''
loss += query_encoder.orthogonal_regularization()
query_encoder = OrthogonalLinear(input_dim=256, embed_dim=256).cuda()

'''

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin  # 负样本对的间隔
    
    def forward(self, pos_z1, pos_z2, neg_z1, neg_z2, pos_ori_1, pos_ori_2, neg_ori_1, neg_ori_2):
        """
        Args:
            pos_z1: [batch_size, embed_dim] 正样本视图1
            pos_z2: [batch_size, embed_dim] 正样本视图2
            neg_z1: [batch_size, embed_dim] 负样本视图1
            neg_z2: [batch_size, embed_dim] 负样本视图2
        Returns:
            loss: 对比损失值
        """
        # 正样本对损失（拉近）
        pos_sim = F.cosine_similarity(pos_z1, pos_z2)  # [batch_size]
        pos_ori_sim1 = F.cosine_similarity(pos_z1, pos_ori_1)
        pos_ori_sim2 = F.cosine_similarity(pos_z2, pos_ori_2)
        neg_ori_sim1 = F.cosine_similarity(neg_z1, neg_ori_1)
        neg_ori_sim2 = F.cosine_similarity(neg_z2, neg_ori_2)
        pos_loss = 10*(1 - pos_sim).mean() + (1-pos_ori_sim1).mean() + (1-pos_ori_sim2).mean() +(1-neg_ori_sim1).mean() + (1-neg_ori_sim2).mean()
        
        # 负样本对损失（推开）
        neg_sim = F.cosine_similarity(neg_z1, neg_z2)  # [batch_size]
        neg_loss = F.relu(neg_sim - self.margin).mean()  # 最小化 max(0, cosine_sim - margin)
        
        # 总损失
        loss = pos_loss + 5*neg_loss
        return loss

class CAVAlignmentModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=256):
        super().__init__()
        # 投影网络
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # 确保激活函数是 GELU
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        # 残差连接投影
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                # 使用 Kaiming 初始化（兼容旧版本）
                nn.init.kaiming_normal_(
                    layer.weight, 
                    mode='fan_out', 
                    nonlinearity='relu'  # 近似代替 GELU
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.kaiming_normal_(
                self.residual_proj.weight, 
                mode='fan_out', 
                nonlinearity='relu'
            )
            if self.residual_proj.bias is not None:
                nn.init.constant_(self.residual_proj.bias, 0.0)
    
    def forward(self, x):
        projected_x = self.projection(x)
        residual_x = self.residual_proj(x)
        return F.normalize(projected_x + residual_x, dim=-1)




class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return x + self.block(x)  # 残差连接

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim=256):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
    
    def forward(self, x):
        return self.net(x) + x



class TCAVLoss(nn.Module):
    """
    改进后的损失函数：层间方差 + 概念中心分离
    """
    def __init__(self, device, num_concepts, var_weight=0.5, similarity_weight=1, margin=0.3):
        super(TCAVLoss, self).__init__()
        self.device = device
        self.var_weight = var_weight      # 层间方差损失权重
        self.similarity_weight = similarity_weight # 概念中心分离权重
        
        # 动态温度参数
        self.temperature = 1.0  
        self.temperature_growth_rate = 1.2  
        self.temperature_max = 1000.0
        
        # 概念中心损失参数
        self.num_concepts = num_concepts
        self.margin = margin  # 不同概念中心最小间距
        # self.centers = nn.Parameter(torch.randn(num_concepts, 1)).to(device)  # 每个概念的 TCAV 目标中心

    def forward(self, input, fused_cav, decoders, class_acts_layer, class_examples, target, mymodel, bottlenecks, concept_labels):
        """
        Args:
            fused_cav: [batch_size, cav_dim] 融合后的概念向量
            concept_labels: [batch_size] 每个样本的概念标签（整数索引）
        """
        # === 动态温度更新 ===
        self.temperature = min(self.temperature * self.temperature_growth_rate, self.temperature_max)

        # === 计算层间 TCAV 方差损失 ===
        tcav_values = []
        similarity_loss = 0
        for layer_idx, bottleneck in enumerate(bottlenecks):
            decoder = decoders[layer_idx]
            reconstructed = decoder(fused_cav)  # [batch_size, cav_dim] → [batch_size, cav_dim]
            cav = reconstructed.requires_grad_(True)
            input_cav = decoder(input[:, layer_idx, :])
            cosine_sim = F.cosine_similarity(input_cav, cav, dim=1) # [batch_size, num_layers]
            sim_loss = (1 - cosine_sim).mean(dim=0).mean()  # 余弦相似度约束
            similarity_loss += sim_loss
            # 计算当前层的 TCAV 值 [batch_size]
            layer_tcav = self.my_compute_tcav(
                mymodel=mymodel, 
                target_class=target, 
                cav=cav, 
                class_acts=class_acts_layer[bottleneck], 
                examples=class_examples, 
                bottleneck=bottleneck,
                device=self.device, 
                temperature=self.temperature
            )
            tcav_values.append(layer_tcav)
        # === 余弦相似度约束项 ===
        similarity_loss /= len(bottlenecks)
        # 层间方差损失 [batch_size] → scalar
        tcav_matrix = torch.stack(tcav_values)  # [num_layers, batch_size]
        consistency_loss = tcav_matrix.var(dim=0).mean()  # 沿层维度计算方差，然后取批次均值
        print("Consistency Loss:", consistency_loss.item(), "Similarity Loss:", similarity_loss.item())
        # import pdb; pdb.set_trace()
        # similarity_loss=similarity_loss.to(consistency_loss.device)
        # === 组合损失 ===
        total_loss = self.var_weight * consistency_loss + self.similarity_weight * similarity_loss

        print("Total Loss:", total_loss)
        return total_loss
    
    def my_compute_tcav(self, mymodel, target_class, cav, class_acts, examples, bottleneck, device, temperature=10):
        """
        计算 TCAV 方向性，并确保计算图不断开（逐样本计算）。
        
        Args:
            mymodel: 目标模型
            target_class: 目标类别
            cav: (torch.Tensor) CAV 向量
            class_acts: (list or torch.Tensor) 每个样本的激活值
            examples: 样本数据
            bottleneck: 指定的层
            device: 计算设备 (cuda 或 cpu)
            
        Returns:
            tcav_score: (torch.Tensor) 方向性得分 (用于反向传播)
        """
        class_id = mymodel.label_to_id(target_class)
        cav = cav.to(device) 
        
        count = 0
        total_samples = len(class_acts)
        
        tcav_scores = [] 
        
        for i in range(total_samples):
            act = np.expand_dims(class_acts[i], 0)
            example = examples[i]
            
            # 计算该样本的梯度
            grad = mymodel.get_gradient(act, [class_id], bottleneck, example)
            grad = torch.tensor(grad, dtype=torch.float32).to(cav.device).view(-1)
            # import pdb; pdb.set_trace()
            # 计算点积
            # dot_prod = torch.dot(grad, cav) 
            cos_sim = F.cosine_similarity(grad.unsqueeze(0), cav, dim=1) 
 
            # 计算方向性，并存入列表
            soft_tcav = torch.sigmoid(-cos_sim * temperature) 
            hard_tcav = (cos_sim<0).float()
            tcav_score = hard_tcav.detach() + (soft_tcav - soft_tcav.detach())
            tcav_scores.append(tcav_score)
        
        tcav_score = torch.stack(tcav_scores).mean(dim=0)  # 计算负向梯度比例
        
        return tcav_score


 
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_layers):
        super(AttentionFusion, self).__init__()
        self.query = nn.Parameter(torch.randn(1, embed_dim))  # Learnable query vector
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, aligned_cavs):
        """
        Args:
            aligned_cavs: Tensor of shape [num_layers, batch_size, embed_dim]
        Returns:
            fused_cav: Tensor of shape [batch_size, embed_dim]
        """
        # Compute keys and values
        keys = self.key(aligned_cavs)        # Shape: [num_layers, batch_size, embed_dim]
        values = self.value(aligned_cavs)   # Shape: [num_layers, batch_size, embed_dim]

        # Compute attention scores
        query = self.query.expand(aligned_cavs.size(1), -1)  # Shape: [batch_size, embed_dim]
        scores = torch.einsum("bd,lbd ->lb", query, keys)  # Shape: [num_layers, batch_size]
        weights = self.softmax(scores)  # Shape: [num_layers, batch_size]

        # Weighted sum of values
        fused_cav = torch.einsum("lb,lbd->bd", weights, values)  # Shape: [batch_size, embed_dim]
        return fused_cav


class TransformerCAVFusion(nn.Module):
    def __init__(self, embedding_dim, num_layers, num_heads=4, ff_dim=256, dropout=0.1):
        super(TransformerCAVFusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Positional encoding for layer indices [num_layers, embedding_dim]
        self.position_encoding = nn.Parameter(torch.randn(num_layers, embedding_dim))

        # Transformer Encoder Components
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True  # 关键修改：启用 batch_first
        )
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, cav_embeddings):
        """
        Args:
            cav_embeddings: [batch_size, num_layers, embedding_dim]
        Returns:
            fused_cav: [batch_size, embedding_dim]
        """
        batch_size = cav_embeddings.size(0)
        
        # 1. 添加位置编码 [num_layers, embedding_dim] → [batch_size, num_layers, embedding_dim]
        position_encoding = self.position_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        cav_embeddings = cav_embeddings + position_encoding

        # 2. 自注意力层（注意 batch_first=True 确保输入为 [batch, seq, dim]）
        attn_output, _ = self.attention_layer(
            query=cav_embeddings, 
            key=cav_embeddings, 
            value=cav_embeddings
        )
        cav_embeddings = self.layer_norm1(cav_embeddings + self.dropout(attn_output))

        # 3. 前馈网络
        ffn_output = self.ffn(cav_embeddings)
        fused_embeddings = self.layer_norm2(cav_embeddings + self.dropout(ffn_output))

        # 4. 沿层维度平均池化 [batch, num_layers, dim] → [batch, dim]
        global_cav = fused_embeddings.mean(dim=1)
        
        return self.output_layer(global_cav)

class CAVAlignTransformer(nn.Module):
    def __init__(self, embedding_size, num_layers, num_transformer_layers=2, nhead=4):
        """
        参数说明：
        - embedding_size: 每个 concept 的嵌入维度，与输入的最后一维对应
        - num_layers: Transformer 处理的序列长度，对应输入的 num_layers
        - num_transformer_layers: TransformerEncoder 中的层数
        - nhead: 多头注意力的头数
        """
        super().__init__()
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        
        # 可选：如果需要投影到 Transformer 的维度，这里假设输入 embedding_size 已经合适
        # self.input_projection = nn.Linear(embedding_size, embedding_size)

        # 添加一个可学习的位置编码参数，形状为 [num_layers, 1, embedding_size]
        # 这样每一层可以获得位置信息（这里假设所有 concept 共用同一位置编码）
        self.positional_encoding = nn.Parameter(torch.zeros(num_layers, 1, embedding_size))
        
        # 构建 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # 可选：后续再添加一个投影头
        # self.projection_head = nn.Linear(embedding_size, embedding_size)
    
    def forward(self, x):
        """
        输入：
            x: tensor，形状为 [num_layers, num_concepts_random, embedding_size]
               这里 num_layers 表示层数，num_concepts_random 表示每层中 concept 数量
        输出：
            out: tensor，形状与输入相同，[num_layers, num_concepts_random, embedding_size]
        """
        # x 的形状： (S, N, E) ，其中 S=num_layers, N=num_concepts_random, E=embedding_size
        # 为每个层的位置添加位置编码，位置编码在 concept 维度上共享
        x = x + self.positional_encoding  # 自动广播到 [num_layers, num_concepts_random, embedding_size]
        
        # 如果需要先投影，可在此处添加：
        # x = self.input_projection(x)
        
        # TransformerEncoder 期望输入形状为 (S, N, E)，此处 x 已符合要求
        out = self.transformer_encoder(x)
        
        # 如果需要再经过投影头，则可添加：
        # out = self.projection_head(out)
        
        return out
    
class CrossLayerContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, lambda_entropy=0.1, lambda_orth=0.1):
        """
        temperature: 对比损失的温度参数
        lambda_entropy: 信息熵正则化的权重
        lambda_orth: 正交正则化的权重
        """
        super().__init__()
        self.temperature = temperature
        self.lambda_entropy = lambda_entropy
        self.lambda_orth = lambda_orth

    def contrastive_loss(self, X):
        """
        计算跨层对比损失，X 的形状为 [num_layers, num_concepts, embedding_size]
        这里假设每一行（每个 concept）在不同层应保持相似。
        """
        num_layers, num_concepts, embedding_size = X.shape
        
        # 归一化得到余弦相似度
        X_norm = F.normalize(X, dim=-1)
        # 计算不同层之间，每个 concept 的相似性
        # 这里通过在层维度上做矩阵乘法获得形状 [num_layers, num_layers, num_concepts]
        similarities = torch.einsum('lij,lkj->lik', X_norm, X_norm)
        
        # 构造 mask，排除相同层（对角线部分），只计算跨层相似性
        layer_mask = (~torch.eye(num_layers, dtype=torch.bool, device=X.device)).unsqueeze(-1)  # [num_layers, num_layers, 1]
        similarities = similarities.masked_fill(~layer_mask, float('-inf'))
        
        # 对每个正样本对（相同 concept 在不同层），我们希望其相似性较高
        # 使用 logsumexp 来计算损失（这里可以根据实际情况调整损失计算方式）
        loss_contrast = - torch.logsumexp(similarities / self.temperature, dim=1).mean()
        return loss_contrast

    def entropy_loss(self, X):
        """
        信息熵正则化，鼓励输出分布多样性，防止塌缩。
        """
        # 将每个向量看作概率分布（通过 softmax）
        p = F.softmax(X, dim=-1)
        entropy = -torch.sum(p * torch.log(p + 1e-8), dim=-1).mean()
        return entropy

    def orthogonality_loss(self, X):
        """
        正交正则化损失，鼓励不同概念之间保持正交性，防止所有向量退化为相同值。
        """
        num_layers, num_concepts, embedding_size = X.shape
        # 将 [num_layers, num_concepts, embedding_size] reshape 为 [num_layers*num_concepts, embedding_size]
        X_reshaped = X.view(num_layers * num_concepts, embedding_size)
        # 计算 Gram 矩阵
        gram = X_reshaped @ X_reshaped.T
        I = torch.eye(gram.size(0), device=X.device)
        loss_orth = torch.norm(gram - I, p='fro')
        return loss_orth

    def forward(self, X):
        loss_contrast = self.contrastive_loss(X)
        loss_entropy = self.entropy_loss(X)
        loss_orth = self.orthogonality_loss(X)
        
        total_loss = loss_contrast + self.lambda_entropy * loss_entropy + self.lambda_orth * loss_orth
        return total_loss, loss_contrast, loss_entropy, loss_orth
    
class MoCoCAV(nn.Module):
    def __init__(self, input_dim, embed_dim, cavs, queue_size=4096, momentum=0.999, temperature=0.07, device = "cuda"):
        super().__init__()
        self.device = device
        self.query_encoder = OrthogonalLinear(input_dim, embed_dim).to(self.device)
        self.key_encoder = OrthogonalLinear(input_dim, embed_dim).to(self.device)
        # Initialize queue
        self.register_buffer("queue", torch.randn(queue_size, embed_dim))
        self.queue = F.normalize(self.queue, dim=1).detach()
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size
        self.cavs = cavs

        self.momentum = momentum
        self.temperature = temperature

    @torch.no_grad()
    def init_queue(self):
        num_layers = len(self.cavs)
        num_concepts = len(self.cavs[0])
        # num_layers = cavs_tensor.size(0)
        # num_concepts = cavs_tensor.size(1)
        self.key_encoder.eval()
        queue_data = []
        for i in range(self.queue_size):
            layer_idx = (i // num_concepts) % num_layers
            concept_idx = i % num_concepts
            # cav = cavs_tensor[layer_idx, concept_idx, :]
            # import pdb; pdb.set_trace()
            cav = self.cavs[layer_idx][concept_idx].unsqueeze(0)
            encoded_cav = F.normalize(self.key_encoder(cav), dim=1).detach()
            queue_data.append(encoded_cav.squeeze(0))

        self.queue = torch.stack(queue_data)
        self.key_encoder.train()

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """update the key encoder by using momentum"""
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        """update queue"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        queue_size = self.queue.size(0)
        assert batch_size <= self.queue.size(0), "queue size is too small to hold the current batch"

        
        # Number of items that can fit at the end of the queue
        remaining_space = queue_size - ptr

        if batch_size <= remaining_space:
            # Case 1: All keys fit in the remaining space
            self.queue[ptr:ptr + batch_size, :] = keys
        else:
            # Case 2: Split the keys into two parts
            self.queue[ptr:, :] = keys[:remaining_space, :]  # Fill the end of the queue
            self.queue[:batch_size - remaining_space, :] = keys[remaining_space:, :]  # Wrap around to the start

        # Update the pointer (modulo to ensure circular behavior)
        ptr = (ptr + batch_size) % queue_size
        self.queue_ptr[0] = ptr




    def forward(self, cav_query, cav_key):
        """
        Args:
            cav_query: 
            cav_key: 
        Returns:
            loss: 
        """
        # normalize the features
        q = F.normalize(self.query_encoder(cav_query), dim=1)
        k = F.normalize(self.key_encoder(cav_key), dim=1)
        k = k.detach()  # no gradient to key encoder
        # compute logits InfoNCE loss
        positive_sim = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)  # positive similarity
        negative_sim = torch.mm(q, self.queue.T)  # negative similarity

        logits = torch.cat([positive_sim, negative_sim], dim=1)  # concatenate positive and negative logits
        logits_with_t = logits / self.temperature  # temperature scaling

        labels = torch.zeros(logits_with_t.size(0), dtype=torch.long).to(self.device)  # positive index is 0
        # loss = F.cross_entropy(logits_with_t, labels)
    

        # 原InfoNCE损失
        ce_loss = F.cross_entropy(logits_with_t, labels)

        # 新增方向对齐损失：最大化正样本对的余弦相似度
        direction_loss = 1 - positive_sim.mean()

        # 总损失 = InfoNCE + λ * 方向损失
        loss = ce_loss + 0.1 * direction_loss


        return loss, k



class IntegrateCAV(nn.Module):
    def __init__(self, cavs, device,concepts_string, autoencoders=None, dim_align_method="zero_padding", num_random_exp=3, save_dir="./analysis"):
        super().__init__()
        self.cavs = cavs
        self.aligned_cavs = []
        self.fused_cavs = []
        self.device = device
        self.__isAligned = False
        self.autoencoders = autoencoders
        self.dim_align_method = dim_align_method
        self.num_random_exp = num_random_exp
        self.save_dir = save_dir
        self.num_concepts = len(cavs[0])//num_random_exp
        self.concepts_string = concepts_string

    def _align_dimension_by_zero(self):
        """
        Align the dimensions of CAVs by zero padding
        """
        cavs_same_dim = []
        max_dim = max([len(cav) for cavs_layer in self.cavs for cav in cavs_layer])
        for cavs_layer in self.cavs:
            cavs_layer_tmp = []
            for cav in cavs_layer:
                if len(cav) < max_dim:
                    cav = np.pad(cav, (0, max_dim - len(cav)), 'constant')
                cav = torch.tensor(cav, dtype=torch.float32).to(self.device)
                cavs_layer_tmp.append(cav)
            cavs_same_dim.append(cavs_layer_tmp)
        return cavs_same_dim, max_dim
    

    def _align_dimension_by_ae(self, type="align"):
        """
        Args:
            autoencoder: trained CAVAutoencoder model
            cavs: cavs for each layer [n_samples, input_dim]
        Returns:
            aligned_cavs: 
        """

        save_dir = os.path.join(self.save_dir,"align_model", self.dim_align_method, self.concepts_string)
        os.makedirs(save_dir, exist_ok=True)
        if not self.autoencoders.overwrite and os.path.exists(os.path.join(save_dir,f"input_cavs_{self.autoencoders.key_params}.npy")):
            print("Input CAVs already exist. Loading from saved files.")
            input_cavs = np.load(os.path.join(save_dir,f"input_cavs_{self.autoencoders.key_params}.npy"), allow_pickle=True)
            if type == "fuse":
                return input_cavs
            tmp=[]
            for layer_cavs in input_cavs:
                layer_cavs_tmp = []
                for cav in layer_cavs:
                    cav = torch.tensor(cav).to(self.device)
                    layer_cavs_tmp.append(cav)
                tmp.append(layer_cavs_tmp)
            print("Input CAVs loaded!")
            return tmp
        input_cavs = []
        for layer_idx, cavs_layer in enumerate(self.cavs):
            cavs_layer_tmp = []
            layer_autoencoder = self.autoencoders.load_autoencoder(layer_idx)
            for cav in cavs_layer:
                cav = torch.tensor(cav).to(self.device)
                cav = layer_autoencoder.module.encode(cav).detach()

                cavs_layer_tmp.append(cav)
            input_cavs.append(cavs_layer_tmp)
            del layer_autoencoder # release memory
        # import pdb; pdb.set_trace()
        save_cavs = []
        for layer_cavs in input_cavs:
            save_layer_cavs = []
            for cav in layer_cavs:
                save_layer_cavs.append(cav.cpu().numpy())
            save_cavs.append(save_layer_cavs)
        # import pdb; pdb.set_trace()
        save_cavs = np.array(save_cavs)
        # import pdb; pdb.set_trace()
        np.save(os.path.join(save_dir,f"input_cavs_{self.autoencoders.key_params}.npy"),  save_cavs)
        print("Input CAVs saved at", os.path.join(save_dir,f"input_cavs_{self.autoencoders.key_params}.npy"))
        if type == "fuse":
            return save_cavs
        return input_cavs

    def align_with_moco(self, queue_size, momentum, temperature, embed_dim=2048, epochs = 2000, overwrite=False):
        save_dir = os.path.join(self.save_dir,"align_model", self.dim_align_method,self.concepts_string)
        if not overwrite :
            if os.path.exists(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy")):
                print("Aligned CAVs already exist. Loading from saved files.")
                self.aligned_cavs = np.load(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"), allow_pickle=True)
                print("Aligned CAVs loaded!")
                self.__isAligned = True
                return self.aligned_cavs
            elif os.path.exists(os.path.join(save_dir,f"query_encoder_{self.autoencoders.key_params}.pth")):
                print("Model already exists. Loading from saved files.")
                model = MoCoCAV(input_dim=embed_dim, embed_dim=embed_dim, cavs=self.cavs, queue_size=queue_size, momentum=momentum, temperature=temperature, device = self.device).to(self.device)
                model.query_encoder.load_state_dict(torch.load(os.path.join(save_dir,f"query_encoder_{self.autoencoders.key_params}.pth")))
                model.query_encoder.eval()
                with torch.no_grad():
                    for cavs_layer in self.cavs:
                        aligned_cavs_layer = []
                        for cav in cavs_layer:
                            cav = torch.tensor(cav).to(self.device)
                            cav = cav.unsqueeze(0)
                            cav = model.query_encoder(cav).squeeze(0)
                            aligned_cavs_layer.append(cav.cpu().numpy())
                        self.aligned_cavs.append(aligned_cavs_layer)
               
                print("CAVs aligned!")
                np.save(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"), self.aligned_cavs)
                print("Aligned CAVs saved at", os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"))
                self.__isAligned = True
                return self.aligned_cavs
        else:
            print("Overwriting the existing files.")
            self.__isAligned = False
            
        
        if self.__isAligned:
            print("CAVs already aligned!")
            return self.aligned_cavs
        

        if self.autoencoders is None:
            raise ValueError("Autoencoders are not provided. Please provide the trained autoencoders.")
        
        if self.dim_align_method == "zero_padding":
            input_cavs, input_dim = self._align_dimension_by_zero()
        elif self.dim_align_method == "autoencoder":
            input_cavs = self._align_dimension_by_ae()
            input_dim = embed_dim
        else:
            raise NotImplementedError(f"Dimension alignment method {self.dim_align_method} is not implemented.")

        # for cavs_layer in input_cavs:
        #     cav_aligned_layer = []
        #     for cav in cavs_layer:
        #         cav_aligned_layer.append(cav.cpu().numpy())
        #     self.aligned_cavs.append(cav_aligned_layer)
        # print("CAVs aligned!")
        # np.save(os.path.join(save_dir,f"aligned_cavs(original)_{self.autoencoders.key_params}.npy"), self.aligned_cavs)
        # print("Aligned CAVs saved at", os.path.join(save_dir,f"aligned_cavs(original)_{self.autoencoders.key_params}.npy"))
        # self.__isAligned = True
        # return self.aligned_cavs
        
        model = MoCoCAV(input_dim=input_dim, embed_dim=embed_dim,cavs=input_cavs, queue_size=queue_size, momentum=momentum, temperature=temperature, device = self.device).to(self.device)
        model.init_queue()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            epoch_loss = 0
            # import pdb; pdb.set_trace()
            cav_query, cav_key = prepare_batch(cavs=input_cavs,device=self.device, batch_size=13)
            # cav_query, cav_key = torch.tensor(cav_query).cuda(), torch.tensor(cav_key).cuda()

            loss, k = model(cav_query, cav_key)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.momentum_update_key_encoder()
            model.dequeue_and_enqueue(k)
            epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        os.makedirs(save_dir, exist_ok=True)
        save_path  = os.path.join(save_dir,f"query_encoder_{self.autoencoders.key_params}.pth")
        torch.save(model.query_encoder.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        model.query_encoder.eval()
        with torch.no_grad():
            for cavs_layer in input_cavs:
                aligned_cavs_layer = []
                for cav in cavs_layer:
                    cav = torch.tensor(cav).to(self.device)
                    # import pdb; pdb.set_trace()
                    cav = cav.unsqueeze(0)
                    cav = model.query_encoder(cav).squeeze(0)
                    aligned_cavs_layer.append(cav.cpu().numpy())
                self.aligned_cavs.append(aligned_cavs_layer)
        print("CAVs aligned!")
        del model # release memory
        self.aligned_cavs = np.array(self.aligned_cavs)
        np.save(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"), self.aligned_cavs)
        print("Aligned CAVs saved at", os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"))
        self.__isAligned = True
        return self.aligned_cavs

    def align_with_transformer(self, overwrite=False):
        save_dir = os.path.join(self.save_dir,"align_model", self.dim_align_method,self.concepts_string)
        input_cavs = self._align_dimension_by_ae(type="fuse")
        input_cavs = torch.tensor(input_cavs).to(self.device)

        #def __init__(self, embedding_size, num_layers, num_transformer_layers=2, nhead=4):
        model = CAVAlignTransformer(
            
            embedding_size=len(input_cavs[0][0]),
            num_layers=len(input_cavs),
            num_transformer_layers=2,
            nhead=4
        ).to(self.device)   
        loss_fn = CrossLayerContrastiveLoss(temperature=0.07, lambda_entropy=0.1, lambda_orth=0.1).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # -------------------------------
        # 4. 训练循环
        # -------------------------------
        num_epochs = 10

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            # 模型前向，输出形状为 [num_layers, num_concepts_random, embedding_size]
            cav_out = model(input_cavs)
            # import pdb; pdb.set_trace()
            # 计算损失
            total_loss, loss_contrast, loss_entropy, loss_orth = loss_fn(cav_out)
            
            # 反向传播与优化
            total_loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss.item():.4f}, "
                    f"Contrast Loss: {loss_contrast.item():.4f}, Entropy Loss: {loss_entropy.item():.4f}, "
                    f"Orth Loss: {loss_orth.item():.4f}")

        os.makedirs(save_dir, exist_ok=True)
        save_path  = os.path.join(save_dir,f"align_transformer_{self.autoencoders.key_params}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        # -------------------------------
        # 5. 测试输出（训练结束后的 CAV）
        # -------------------------------
        model.eval()
        with torch.no_grad():
            self.aligned_cavs= model(input_cavs)

        np.save(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"), self.aligned_cavs)
        print("Aligned CAVs saved at", os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}.npy"))
        self.__isAligned = True
        return self.aligned_cavs

  
    def prepare_batch(self, input_cavs):
        anchors = []
        positives = []
        negatives = []
        
        # 遍历所有概念和层

        for layer_idx in range(len(input_cavs)):
            # 当前层中该概念的所有随机实验 CAV
            start_idx = 0
            end_idx = self.num_random_exp
            concept_cavs = input_cavs[layer_idx][start_idx:end_idx]
            
            # 为每个 CAV 生成跨层正样本对和同层负样本
            for exp_idx in range(self.num_random_exp):
                # 锚点：当前层的 CAV
                anchor = concept_cavs[exp_idx]
                
                # === 正样本：其他层同一概念的 CAV ===
                # 选择其他层
                # other_layers = [l for l in range(len(input_cavs)) if l != layer_idx]
                # target_layer = np.random.choice(other_layers)
                target_layer = len(input_cavs) - layer_idx -1
                # 同一概念的随机实验
                # import  pdb; pdb.set_trace()
                target_exp = np.random.randint(0, self.num_random_exp)
                positive = input_cavs[target_layer][target_exp]
                
                # === 负样本：同一层不同概念的 CAV ===
                neg_samples = []
                for i in range(len(input_cavs[1])-self.num_random_exp):
                    neg_cav = input_cavs[layer_idx][self.num_random_exp + i]
                    neg_samples.append(neg_cav)
                
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(neg_samples)
        
        # 转换为 Tensor 并移动到设备
        anchors = torch.tensor(np.array(anchors), dtype=torch.float32).to(self.device)
        positives = torch.tensor(np.array(positives), dtype=torch.float32).to(self.device)
        negatives = torch.tensor(np.array(negatives), dtype=torch.float32).to(self.device)
        
        return anchors, positives, negatives
    def contrastive_loss(self, z1, z2, temperature=0.05):
        """改进的对比损失，显式处理正负样本"""
        batch_size = z1.size(0)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(z1, z2.T) / temperature  # [B, B]
        
        # 正样本对位于对角线
        positive_sim = torch.diag(sim_matrix)
        
        # 负样本对为所有非对角线元素
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
        negative_sim = sim_matrix[mask].view(batch_size, -1)
        
        # 构造logits
        logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)
        
        # 计算交叉熵损失
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, labels)
    
    def train(self,embed_dim=2048, epochs=1000, batch_size=256, lr=1e-3, overwrite=False):#  embed_dim=2048, epochs = 2000):
        save_dir = os.path.join(self.save_dir,"align_model", self.dim_align_method, self.concepts_string)
        if not overwrite :
            if os.path.exists(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}_{k1}_{k2}.npy")) and fuse_input == "aligned_cavs":
                print("Aligned CAVs already exist. Loading from saved files.")
                self.aligned_cavs = np.load(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}_{k1}_{k2}.npy"), allow_pickle=True)
                print("Aligned CAVs loaded!")
                self.__isAligned = True
                return self.aligned_cavs
            elif os.path.exists(os.path.join(save_dir,f"input_cavs_{self.autoencoders.key_params}.npy")) and fuse_input == "input_cavs":
                print("Input CAVs already exist. Loading from saved files.")
                self.aligned_cavs = np.load(os.path.join(save_dir,f"input_cavs_{self.autoencoders.key_params}.npy"), allow_pickle=True)
                print("Input CAVs loaded!")
                self.__isAligned = False
                return self.aligned_cavs
                
        model = CAVAlignmentModel(input_dim=embed_dim, hidden_dim=4096, output_dim=embed_dim).to(self.device) # , input_dim=2048, hidden_dim=1024, output_dim=256

        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        input_cavs = self._align_dimension_by_ae(type="fuse")
        # import pdb; pdb.set_trace()
        # input_cavs = normalize_cav(input_cavs)
        # import pdb; pdb.set_trace()
        # 准备所有样本对
        # num_neg_per_pos=5
        # pos_pairs, neg_pairs = self.prepare_pairs(input_cavs=input_cavs,num_neg_per_pos=num_neg_per_pos)
        # all_pairs = pos_pairs + neg_pairs

        # contrastive_loss = ContrastiveLoss(margin=0.15)
    
        # 优化器增强：添加学习率预热和余弦退火
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        # warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, 
        #     lr_lambda=lambda epoch: min(1.0, (epoch + 1) / 100)  # 100步预热
        # )
        
        # 梯度裁剪
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
        # 初始化 InfoNCE 损失函数
        info_nce_loss = InfoNCELoss(temperature=0.1)
        consistency_loss = ConsistencyLoss()
        for epoch in range(epochs):
            # 生成批次数据
            anchors, positives, negatives = self.prepare_batch(
                input_cavs, 
            )
            # import pdb; pdb.set_trace()
            # 将数据移动到设备
            anchors = anchors.to(self.device)
            positives = positives.to(self.device)
            negatives = negatives.to(self.device)
            
            # 前向传播
            anchor_emb = model(anchors)       # [batch_size, embed_dim]
            positive_emb = model(positives)   # [batch_size, embed_dim]
            negative_emb = model(negatives)   # [batch_size, num_neg, embed_dim]
            
            # 计算 InfoNCE 损失
            loss_info = info_nce_loss(anchor_emb, positive_emb, negative_emb)
            loss_con = consistency_loss(anchor_emb, positive_emb, anchors, positives)
            loss = nce_loss * loss_info + con_loss * loss_con # default con_loss =  3 nce_loss = 1
            # print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            # print(f"Epoch [{epoch+1}/{epochs}], InfoNCE Loss: {loss_info.item():.4f}")
            # print(f"Epoch [{epoch+1}/{epochs}], Consistency Loss: {loss_con.item():.4f}")
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 打印损失
            # if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            print(f"Epoch [{epoch+1}/{epochs}], InfoNCE Loss: {loss_info.item():.4f}")
            print(f"Epoch [{epoch+1}/{epochs}], Consistency Loss: {loss_con.item():.4f}")


        aligned_cavs=[]
        model.eval()
        proj_cavs = input_cavs[:,:10,:]
        with torch.no_grad():
            for cavs_layer in proj_cavs:
                aligned_cavs_layer = []
                for cav in cavs_layer:
                    cav = torch.tensor(cav).to(self.device)
                    # import pdb; pdb.set_trace()
                    cav = cav.unsqueeze(0)
                    cav = model(cav).squeeze(0)
                    aligned_cavs_layer.append(cav.cpu().numpy())
                aligned_cavs.append(aligned_cavs_layer)
        print("CAVs aligned!")
        self.aligned_cavs = np.array(aligned_cavs)
        del model # release memory
        np.save(os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}_{k1}_{k2}.npy"), self.aligned_cavs)
        print("Aligned CAVs saved at", os.path.join(save_dir,f"aligned_cavs_{self.autoencoders.key_params}_{k1}_{k2}.npy"))
        self.__isAligned = True
        return self.aligned_cavs

    
    def fuse(self,fuse_method, bottlenecks,concepts, target, num_random_exp, overwrite=False):
        """
        Fuse the aligned CAVs
        """
        if not self.__isAligned:
            print("WARNING: CAVs are not aligned. Please align the CAVs first.")
            # raise ValueError("CAVs are not aligned. Please align the CAVs first.")

        save_dir = os.path.join(self.save_dir,"fuse_model", self.dim_align_method, fuse_method, self.concepts_string, fuse_input)
        if not overwrite:
            if os.path.exists(os.path.join(save_dir,f"fused_cavs_{self.autoencoders.key_params}.npy")):
                print("Fused CAVs already exist. Loading from saved files.")
                self.fused_cavs = np.load(os.path.join(save_dir,f"fused_cavs_{self.autoencoders.key_params}.npy"), allow_pickle=True)
                print("Fused CAVs loaded!")
                return self.fused_cavs
        
            
        if fuse_method == "mean":
            fused_cavs = np.mean(np.array(self.aligned_cavs), axis=0)
        elif fuse_method == "attention":
            attention_fusion = AttentionFusion(embed_dim=len(self.aligned_cavs[0][0]), num_layers=len(self.aligned_cavs)).to(self.device)
            aligned_cavs = torch.tensor(self.aligned_cavs, dtype=torch.float32).to(self.device)
            fused_cavs = attention_fusion(aligned_cavs).cpu().detach().numpy()
        elif fuse_method == "weight":
            # import pdb; pdb.set_trace()
            # variances = [np.var(embed) for embed in self.aligned_cavs]
            fused_cavs = []
            planes = [self.aligned_cavs[:, i, :] for i in range(len(self.aligned_cavs[0]))]
            planes = np.array(planes)
            for i in range(len(planes)):
                variance = [np.var(embed) for embed in planes[i]]
                weight = [var / sum(variance) for var in variance]
                fused_cav = np.sum([weight * embed for weight, embed in zip(weight, planes[i])], axis=0)
                fused_cavs.append(fused_cav)
            
            fused_cavs = np.array(fused_cavs)
        elif fuse_method == "cosine":
            input_cavs = self.aligned_cavs
            fused_cavs = []
            for concept_idx in range(len(input_cavs[0])):
                cos_sim = []
                for layer_idx in range(len(input_cavs)):
                    cos_sim.append(cosine_similarity(input_cavs[layer_idx][concept_idx],self.aligned_cavs[layer_idx][concept_idx])+1)# add 1 to avoid negative value
                weight = [cos/sum(cos_sim) for cos in cos_sim]
                fused_cav = np.sum([weight * embed for weight, embed in zip(weight, self.aligned_cavs[:, concept_idx,:])], axis=0)
                fused_cavs.append(fused_cav)
            fused_cavs = np.array(fused_cavs)
        elif fuse_method == "transformer":
            import tcav.activation_generator as act_gen
            if fuse_input == "input_cavs":
                print("using input cavs as input")
                aligned_cavs = self._align_dimension_by_ae(type="fuse")
            else:
                print("using aligned_cavs as input")
                aligned_cavs = self.aligned_cavs
            # import pdb; pdb.set_trace()
            if not isinstance(aligned_cavs, torch.Tensor):
                aligned_cavs = torch.from_numpy(aligned_cavs)

            aligned_cavs = aligned_cavs[:, :num_random_exp, :].float().to(self.device)

            # aligned_cavs = aligned_cavs[:,:num_random_exp,:]
            # aligned_cavs = aligned_cavs[:,:10,:]
            # import pdb; pdb.set_trace()
            
            act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=100)
            decoders = []
            for layer_idx, _ in enumerate(bottlenecks):
                decoder = self.autoencoders.load_autoencoder(layer_idx).module.decode
                decoders.append(decoder)

            class_examples = act_generator.get_examples_for_concept(target) # get examples for target class(not sure)
            class_acts_layer = {}
            acts = act_generator.process_and_load_activations(bottlenecks, concepts + [target])
            for bottleneck in bottlenecks:
                acts_instance = acts[target][bottleneck]
                class_acts_layer[bottleneck] = acts_instance
           
            label_concepts = []
            for concept_idx, _ in enumerate(concepts):
                for _ in range(num_random_exp):
                    label_concepts.append(concept_idx)

            # model = TransformerCAVFusion(embedding_dim=len(aligned_cavs[0][0]), num_layers=len(aligned_cavs)).to(self.device)
            model = TransformerCAVFusion(embedding_dim=len(aligned_cavs[0][0]), num_layers=len(aligned_cavs))
            # model = nn.DataParallel(model,device_ids=[0,1])
            # model = nn.DataParallel(model,device_ids=[3,2])
            model = model.to(self.device)
            cav_batchs = [aligned_cavs[:, i, :] for i in range(len(aligned_cavs[0]))]
            # import pdb; pdb.set_trace()
            if len(label_concepts) != len(cav_batchs):
                raise ValueError("Number of concepts does not match the number of CAVs.")
            dataset = CAVDataset(cav_batchs, label_concepts)

            num_epochs = 30
            model.train()
            # device, num_concepts, var_weight=0.5, center_weight=1, margin=0.3
            criterion = TCAVLoss(
                device=self.device,
                num_concepts=len(concepts),
                var_weight=var_loss,   # default = 3
                similarity_weight=sim_loss,
                margin=0.15
            )

            optimizer = optim.Adam(model.parameters(), lr=5e-4)
            scaler = torch.cuda.amp.GradScaler()
            for epoch in range(num_epochs):
                total_loss = 0  
                for cav_batch, labels in DataLoader(dataset, batch_size=4, shuffle=True):
                    
                    cav_batch = cav_batch.to(self.device)
                    optimizer.zero_grad()
                    # with torch.cuda.amp.autocast():  # 开启混合精度
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):

                        fused_cav = model(cav_batch)
                        loss = criterion(
                            input=cav_batch.float(),
                            fused_cav=fused_cav.float(),
                            decoders=decoders, 
                            class_acts_layer=class_acts_layer, 
                            class_examples=class_examples, 
                            target=target, 
                            mymodel=mymodel, 
                            bottlenecks=bottlenecks,
                            concept_labels=labels
                        ) 

                    scaler.scale(loss).backward()  # 使用 Scaler 进行反向传播
                    scaler.step(optimizer)  # 更新优化器
                    scaler.update()  # 更新 Scaler
                    total_loss += loss.item()

                    # fused_cav = model(cav_batch)
                    # fused_cav.requires_grad_(True)
                    # loss = criterion(
                    #     input=cav_batch,
                    #     fused_cav=fused_cav,
                    #     decoders=decoders, 
                    #     class_acts_layer=class_acts_layer, 
                    #     class_examples=class_examples, 
                    #     target=target, 
                    #     mymodel=mymodel, 
                    #     bottlenecks=bottlenecks,
                    #     concept_labels=labels
                    # )  # 按层平均
                    # loss.backward()
                    # optimizer.step()
                    # total_loss += loss.item()

                print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss:.4f}")

            model = model.eval()
            os.makedirs(save_dir, exist_ok=True)
            save_path  = os.path.join(save_dir,f"fuse_model_{self.autoencoders.key_params}_{k2}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

            fused_cavs = []
            for cav_embeddings in cav_batchs:
                cav_tensor = torch.tensor(cav_embeddings).to(self.device)
                cav_tensor = cav_tensor.unsqueeze(0)
                fused_cav = model(cav_tensor).cpu().detach().numpy()
                fused_cav = np.squeeze(fused_cav) 
                fused_cavs.append(fused_cav)

        else:
            raise NotImplementedError(f"Fuse method {fuse_method} is not implemented.")
        self.fused_cavs = fused_cavs # [num_concepts, cav_dim]
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir,f"fused_cavs_{self.autoencoders.key_params}_{k1}_{k2}.npy"), fused_cavs)
        print("Fused CAVs saved at", os.path.join(save_dir,f"fused_cavs_{self.autoencoders.key_params}_{k1}_{k2}.npy"))
        return self.fused_cavs
