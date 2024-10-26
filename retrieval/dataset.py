import torch.utils.data
import pandas as pd


def custom_collate_fn(batch):
    A, mean_pooling_vec, retrieved_label_list, retrieved_textual_feature_embedding, label = zip(*batch)

    label = [int(l) for l in label]
    A = torch.stack(list(A), dim=0)

    return (A,
            torch.tensor(mean_pooling_vec, dtype=torch.float32),
            torch.tensor(retrieved_label_list, dtype=torch.float32),
            torch.tensor(retrieved_textual_feature_embedding, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32).unsqueeze(-1))


class MyData(torch.utils.data.Dataset):

    def __init__(self, retrieval_num, path):
        super().__init__()

        self.path = path
        self.retrieval_num = retrieval_num
        self.dataframe = pd.read_pickle(path)
        self.A = self.dataframe['A']
        self.label = self.dataframe['Label']
        self.mean_pooling_vec = self.dataframe['mean_pooling_vec']
        self.retrieved_label_list = self.dataframe['retrieved_label']
        self.retrieved_textual_feature_embedding = self.dataframe['retrieved_textual_feature_embedding']

    def __getitem__(self, item):
        retrieval_num = int(self.retrieval_num)
        label = self.label[item]
        mean_pooling_vec = self.mean_pooling_vec[item]
        retrieved_label_list = (self.retrieved_label_list[item])[:retrieval_num]
        retrieved_textual_feature_embedding = (self.retrieved_textual_feature_embedding[item])[:retrieval_num]
        A = self.A[item]

        return A, mean_pooling_vec, retrieved_label_list, retrieved_textual_feature_embedding, label

    def __len__(self):
        return len(self.dataframe)
