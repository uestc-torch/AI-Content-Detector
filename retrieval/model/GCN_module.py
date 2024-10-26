import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GCN(nn.Module):

    def __init__(self, retrieval_num=50, feature_dim=768, hidden_dim=512, num_gcn_layers=3):
        super(GCN, self).__init__()
        self.feature_dim = feature_dim
        self.retrieval_num = retrieval_num
        self.hidden_dim = hidden_dim

        # 图卷积层
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(nn.Linear(feature_dim, feature_dim))

        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 2)
        )

        self.label_mlp= nn.Linear(retrieval_num,feature_dim)



    def forward(self, A, mean_pooling_vec, retrieved_label_list, retrieved_textual_feature_embedding):
        """
        参数:
        mean_pooling_vec: torch.Size([256, 1, 768]) - 输入特征向量
        retrieved_label_list: torch.Size([256, 50]) - 检索的标签
        retrieved_textual_feature_embedding: torch.Size([256, 50, 1, 768]) - 检索的特征向量

        返回:
        output: torch.Size([256, 1]) - 预测概率
        """
        mean_pooling_vec = mean_pooling_vec.unsqueeze(1)
        retrieved_textual_feature_embedding = retrieved_textual_feature_embedding.unsqueeze(2)
        retrieval_num = self.retrieval_num

        # Step 1: 构建邻接矩阵A (51x51)，包含 mean_pooling_vec 和 50 个检索元素的完全连接图
        A = A  # (256, 51, 51)

        matrix = np.zeros((51, 51), dtype=np.float32)
        matrix[:, 0] = 1
        matrix[0, :] = 1
        batch_size = A.size(0)
        A = torch.tensor(matrix).unsqueeze(0).repeat(batch_size, 1, 1).to(A.device)

        # Step 2: 拼接mean_pooling_vec与retrieved_textual_feature_embedding
        mean_pooling_vec_expanded = mean_pooling_vec.squeeze(1).unsqueeze(1)  # (256, 1, 768)
        node_features = torch.cat([mean_pooling_vec_expanded, retrieved_textual_feature_embedding.squeeze(2)],
                                  dim=1)  # (256, 51, 768)

        # Step 3: 传入图卷积网络
        for gcn_layer in self.gcn_layers:
            # 对邻接矩阵A进行矩阵乘法来传播信息
            node_features = torch.matmul(A, node_features)  # (256, 51, 768)
            node_features = gcn_layer(node_features)  # 图卷积
            node_features = F.relu(node_features)  # 激活函数

        # Step 4: 提取mean_pooling_vec对应的节点特征 (假设第一个节点是 mean_pooling_vec)
        output_features = node_features[:, 0, :]  # (256, 768*2)

        label_features = self.label_mlp(retrieved_label_list.float())
        # Step 5: 通过MLP映射到预测概率
        output_features= torch.cat([output_features, label_features], dim=-1)  # (256, 768*2+50)

        output = self.mlp(output_features)  # (256, 2)

        # Step 6: 使用sigmoid归一化
        # output = torch.sigmoid(output)  # (256, 1)

        return output, output_features


if __name__ == '__main__':
    # 构造模型
    model = GCN(retrieval_num=50, feature_dim=768, hidden_dim=512, num_gcn_layers=3)
    print(model)

    # 构造输入
    mean_pooling_vec = torch.randn(256, 1, 768)
    retrieved_label_list = torch.randint(0, 2, (256, 50))
    retrieved_textual_feature_embedding = torch.randn(256, 50, 1, 768)

    # 模型预测
    output = model(mean_pooling_vec, retrieved_label_list, retrieved_textual_feature_embedding)
    print(output.size())

