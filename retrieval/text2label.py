import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import ast



def text2vec(model_path, text_list, batch_size=32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)

    def get_embeddings(text_list, batch_size=32):
        embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size), desc="Generating Embeddings"):
            batch_texts = text_list[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)
    
    embeddings = get_embeddings(text_list, batch_size=32)
    return embeddings

def retrieve_item(q_vecs, re_vecs):
    q_norms = q_vecs.norm(dim=1, keepdim=True)
    re_norms = re_vecs.norm(dim=1, keepdim=True)
    sim_scores = torch.matmul(q_vecs, re_vecs.transpose(0, 1)) / (q_norms * re_norms.t())
    return sim_scores

def retrieval(query_data_vec, base_data_vec, base_data_label, batch_size=32):

    retrieve_vecs = torch.stack([torch.tensor(vec) for vec in base_data_vec])

    def process_data(data):
        retrieved_text_vec = []
        retrieved_label = []

        for i in tqdm(range(0, len(data), batch_size)):

            batch_q_vecs = torch.stack([torch.tensor(vec) for vec in query_data_vec[i:i+batch_size]])
            
            sim_scores = retrieve_item(batch_q_vecs, retrieve_vecs)
            _, top_k_ids = torch.topk(sim_scores, 51, dim=-1)

            for _, top_k_id in enumerate(top_k_ids):
                top_k_id = top_k_id.tolist()
                retrieved_text_vec.append([base_data_vec[j] for j in top_k_id])
                retrieved_label.append([float(base_data_label[j]) for j in top_k_id])
        return retrieved_text_vec, retrieved_label

    retrieved_text_vec, retrieved_label = process_data(query_data_vec)
    return retrieved_text_vec, retrieved_label

def predict(model_path, A, mean_pooling_vec, retrieved_label_list, retrieved_textual_feature_embedding):

    model = torch.load(model_path)
    model.eval()

    A = torch.stack(list(A)).to('cuda:0')
    mean_pooling_vec = torch.tensor(mean_pooling_vec, dtype=torch.float32).to('cuda:0')
    retrieved_label_list = torch.tensor(retrieved_label_list, dtype=torch.float32).to('cuda:0')
    # 先转换为 numpy 数组
    retrieved_textual_feature_embedding = np.array(retrieved_textual_feature_embedding)

    # 然后再转换为 PyTorch tensor
    retrieved_textual_feature_embedding = torch.tensor(retrieved_textual_feature_embedding, dtype=torch.float32).to('cuda:0')
    retrieved_label_list = retrieved_label_list[:,1:]
    retrieved_textual_feature_embedding = retrieved_textual_feature_embedding[:,1:,:]

    # print(
    #     f"A: {A.shape}\n"
    #     f"mean_pooling_vec: {mean_pooling_vec.shape}\n"
    #     f"retrieved_label_list: {retrieved_label_list.shape}\n"
    #     f"retrieved_textual_feature_embedding: {retrieved_textual_feature_embedding.shape}\n"
    # )
                                                       

    with torch.no_grad():

        output, output_features = model.forward(A, mean_pooling_vec, retrieved_label_list, retrieved_textual_feature_embedding)
        output = output.to('cpu').numpy()
        output = np.where(output > 0.5, 1, 0)

    return output, output_features

def get_matrix(retrieval_num , retrieved_label_list_batch):

    batch_size = 32
    retrieval_num = 50

    A_batch = torch.zeros(batch_size, retrieval_num + 1, retrieval_num + 1)

    for b in range(batch_size):
        A = torch.zeros(retrieval_num + 1, retrieval_num + 1)

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

def process_matrix(retrieved_label_list, retrieval_num=50):

    batch_size = 32

    retrieved_label_list = torch.tensor(retrieved_label_list)

    A_list = []
    for i in tqdm(range(0, len(retrieved_label_list), batch_size)):
        batch_retrieved_label_list = retrieved_label_list[i:i+batch_size]
        A = get_matrix(retrieval_num, batch_retrieved_label_list)
        A_list.extend(A)
    return A_list

def origin_dataset_process(txt_path, train_path, valid_path, test_path, dataset_path):

    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        entry = ast.literal_eval(line.strip())
        data.append(entry)

    df = pd.DataFrame(data)
    df = df[['ID', 'Label', 'Content']]
    df.to_pickle(dataset_path)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df.to_pickle(train_path)
    val_df.to_pickle(valid_path)
    test_df.to_pickle(test_path)

def test(train_data_path, valid_data_path, test_data_path):

    train_data = pd.read_pickle(train_data_path)
    valid_data = pd.read_pickle(valid_data_path)
    test_data = pd.read_pickle(test_data_path)

    train_text = train_data['Content'].tolist()
    valid_text = valid_data['Content'].tolist()
    test_text = test_data['Content'].tolist()

    train_label = train_data['Label'].tolist()
    valid_label = valid_data['Label'].tolist()

    base_text = train_text + valid_text
    base_label = train_label + valid_label

    query_text = test_text

    return base_text, base_label, query_text

if __name__ == "__main__":
    # 输入路径
    origin_txt_path = r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/train_data_32k.txt'
    bert_path = r'/home/icdm/CodeSpace/nlp_test/pretrained_best'
    model_path = r'/home/icdm/CodeSpace/nlp_test/retrieval/result/train_graph_attention_text_classi_50_MSE_2024-10-26_16-34-30/trained_model/model_17.pth'


    train_path = r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/train.pkl'
    valid_path = r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/valid.pkl'
    test_path = r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/test.pkl'
    dataset_path = r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/dataset.pkl'
    origin_dataset_process(origin_txt_path, train_path, valid_path, test_path, dataset_path)
    base_text, base_label, query_text = test(train_path, valid_path, test_path)

    base_embeddings = text2vec(bert_path, base_text, batch_size=32)
    query_embeddings = text2vec(bert_path, query_text, batch_size=32)
    print('Text to vector is done!')

    retrieved_text_vec, retrieved_label = retrieval(query_embeddings, base_embeddings, base_label, batch_size=32)
    print('Retrieval is done!')

    A_list = process_matrix(retrieved_label, retrieval_num=50)
    print('Matrix is done!')

    output, output_features = predict(model_path, A_list, query_embeddings, retrieved_label, retrieved_text_vec)
    predict = output.argmax(axis=-1)
    accuracy = (predict == np.array(base_label)).mean()
    print("accuracy:" ,accuracy)
    print(output)         
    print(output_features)  # (bs, 768*2+50)














