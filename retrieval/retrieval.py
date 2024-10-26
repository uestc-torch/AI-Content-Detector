import pandas as pd
import torch
from tqdm import tqdm
import ast
from sklearn.model_selection import train_test_split

def get_matrix(retrieval_num , retrieved_label_list_batch):

    batch_size = 32
    retrieval_num = 50

    A_batch = torch.zeros(batch_size, retrieval_num + 1, retrieval_num + 1)

    for b in range(batch_size):
        A = torch.zeros(retrieval_num + 1, retrieval_num + 1)

        # 所有特征向量与检索向量连一条边
        A[0, 1:] = 1
        A[1:, 0] = 1

        retrieved_label_list = retrieved_label_list_batch[b]
        for i in range(1, retrieval_num + 1):
            for j in range(i + 1, retrieval_num + 1):
                if retrieved_label_list[i - 1] == retrieved_label_list[j - 1]:
                    A[i, j] = 1
                    A[j, i] = 1

        A_batch[b] = A

    return A_batch

def process_matrix(dataset_path, retrieval_num=50):

    dataset = pd.read_pickle(dataset_path)

    batch_size = 32

    retrieved_label_list = dataset['retrieved_label'].tolist()
    mean_pooling_vec = dataset['mean_pooling_vec'].tolist()

    retrieved_label_list = torch.tensor(retrieved_label_list)
    mean_pooling_vec = torch.tensor(mean_pooling_vec)

    A_list = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_retrieved_label_list = retrieved_label_list[i:i+batch_size]
        A = get_matrix(retrieval_num, batch_retrieved_label_list)
        A_list.extend(A)

    dataset['A'] = A_list

    dataset.to_pickle(dataset_path)

    return dataset



def retrieve_item(q_vecs, re_vecs):
    q_norms = q_vecs.norm(dim=1, keepdim=True)
    re_norms = re_vecs.norm(dim=1, keepdim=True)
    sim_scores = torch.matmul(q_vecs, re_vecs.transpose(0, 1)) / (q_norms * re_norms.t())
    return sim_scores

def retrieval(train_path, valid_path, test_path, batch_size=32):

    train_data = pd.read_pickle(train_path)
    valid_data = pd.read_pickle(valid_path)
    test_data = pd.read_pickle(test_path)

    retrieval_data = pd.concat([train_data, valid_data], axis=0)
    retrieval_data.reset_index(drop=True, inplace=True)

    retrieve_vecs = torch.stack([torch.tensor(vec) for vec in retrieval_data['mean_pooling_vec']])

    def process_data(data):
        retrieved_text_vec = []
        retrieved_label = []

        for i in tqdm(range(0, len(data), batch_size)):
            batch_q_vecs = torch.stack([torch.tensor(vec) for vec in data['mean_pooling_vec'][i:i+batch_size]])
            
            sim_scores = retrieve_item(batch_q_vecs, retrieve_vecs)
            _, top_k_ids = torch.topk(sim_scores, 51, dim=-1)

            for idx, top_k_id in enumerate(top_k_ids):
                top_k_id = top_k_id.tolist()
                top_k_id = top_k_id[1:]
                retrieved_text_vec.append([retrieval_data['mean_pooling_vec'][j] for j in top_k_id])
                retrieved_label.append([float(retrieval_data['Label'][j]) for j in top_k_id])

        data['retrieved_textual_feature_embedding'] = retrieved_text_vec
        data['retrieved_label'] = retrieved_label
        return data

    train_data = process_data(train_data)
    valid_data = process_data(valid_data)
    test_data = process_data(test_data)

    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    train_data.to_pickle(train_path)
    valid_data.to_pickle(valid_path)
    test_data.to_pickle(test_path)

    return train_data, valid_data, test_data


if __name__ == '__main__':
    
    train_path = r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/train.pkl'
    valid_path = r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/valid.pkl'
    test_path = r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/test.pkl'

    retrieval(train_path, valid_path, test_path)
    print('Retrieval finished!')

    process_matrix(train_path)
    print('Train matrix processed!')
    process_matrix(valid_path)
    print('Valid matrix processed!')
    process_matrix(test_path)
    print('Test matrix processed!')

