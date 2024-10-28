import json
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import re
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from gat_train import main as GAT_main
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
    retrieved_textual_feature_embedding = np.array(retrieved_textual_feature_embedding)

    retrieved_textual_feature_embedding = torch.tensor(retrieved_textual_feature_embedding, dtype=torch.float32).to('cuda:0')
    retrieved_label_list = retrieved_label_list[:,1:]
    retrieved_textual_feature_embedding = retrieved_textual_feature_embedding[:,1:,:]                                                  

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

def test(train_data_path, test_data_path):

    train_data = pd.read_pickle(train_data_path)
    test_data = pd.read_pickle(test_data_path)

    train_text = train_data['Content'].tolist()
    test_text = test_data['Content'].tolist()

    train_label = train_data['Label'].tolist()

    base_text = train_text
    base_label = train_label

    query_text = test_text

    return base_text, base_label, query_text


# 1. 解析txt文件为字典列表
def parse_txt_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        pattern = re.compile(r"{'ID': '(\d+)', 'Label': '(\d+)', 'Content': '(.+?)'}", re.S)
        matches = pattern.findall(content)
        for match in matches:
            entry = {
                'ID': match[0],
                'labels': match[1],
                'Content': match[2].strip()
            }
            data.append(entry)
    return data


# 2. 保存模型和训练状态的函数
def save_model_and_tokenizer(model, tokenizer, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


# 3. 加载模型和分词器的函数
def load_model_and_tokenizer(model_dir):
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    return model, tokenizer




def compute_metrics(pred):
        predictions, labels = pred
        predictions = torch.tensor(predictions)
        preds = torch.argmax(predictions, dim=1)
        labels = torch.tensor(labels)
        accuracy = (preds == labels).float().mean().item()
        return {"accuracy": accuracy  }

def compute_metrics_with_prob(pred):
    predictions, labels = pred
    predictions = torch.tensor(predictions)
    probs = torch.softmax(predictions, dim=1)  # 计算概率
    preds = torch.argmax(predictions, dim=1)
    labels = torch.tensor(labels)

    # 计算准确率
    accuracy = (preds == labels).float().mean().item()

    # 将结果存储为字典列表的形式，以包含 ID, Label, Prob
    eval_results = []
    for i, (pred_label, true_label, prob) in enumerate(zip(preds, labels, probs)):
        result = {
            "ID": val_dataset[i]['ID'],  # 从val_dataset中获取ID
            "Label": int(pred_label.item()),  # 预测的标签
            "Pred": prob.argmax(-1).item(),
            "Confidence": prob.max().item()  # 预测的概率
        }
        eval_results.append(result)

    # 将eval_results列表转换为DataFrame并保存或输出
    eval_results_df = pd.DataFrame(eval_results)
    eval_results_df.to_csv("evaluation_results.csv", index=False, encoding='utf-8')
    print("Evaluation results saved to evaluation_results.csv")
    accuracy = (preds == labels).float().mean().item()
    return {"accuracy": accuracy}


def pretrain_model(train_dataset, val_dataset):
        
    # 加载模型和分词器
    model_name = "/home/icdm/CodeSpace/nlp_test/roberta-base/"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 对content部分做tokenizer
    def tokenize_function(examples):
        return tokenizer(examples['Content'], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    # 训练参数设置
    training_args = TrainingArguments(
        output_dir="/home/icdm/CodeSpace/nlp_test/results",
        evaluation_strategy="epoch",    # 每个epoch评估
        save_strategy="epoch",          # 每个epoch保存一次
        load_best_model_at_end=True,    # 在训练结束时加载最佳模型
        metric_for_best_model="accuracy",  # 使用accuracy判断最佳模型
        greater_is_better=True,         # accuracy 越高越好
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=30,
        weight_decay=0.01,
        logging_dir="nlp_test/logs",
        logging_steps=10,
        save_total_limit=1,             # 只保留一个最佳模型
    )
    # 评估指标计算
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    # 训练模型
    trainer.train()


def model_test(train_dataset, test_dataset):
    model_dir = "/home/icdm/CodeSpace/nlp_test/pretrained_best"
    loaded_model, loaded_tokenizer = load_model_and_tokenizer(model_dir)
    # 打印最佳模型的评估结果
    def tokenize_function(examples):
        return loaded_tokenizer(examples['Content'], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
            output_dir="/home/icdm/CodeSpace/nlp_test/results",
            evaluation_strategy="epoch",    # 每个epoch评估
            save_strategy="epoch",          # 每个epoch保存一次
            load_best_model_at_end=True,    # 在训练结束时加载最佳模型
            metric_for_best_model="accuracy",  # 使用accuracy判断最佳模型
            greater_is_better=True,         # accuracy 越高越好
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=30,
            weight_decay=0.01,
            logging_dir="nlp_test/logs",
            logging_steps=10,
            save_total_limit=1,             # 只保留一个最佳模型
        )

    tester = Trainer(
        model=loaded_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=loaded_tokenizer,
        compute_metrics=compute_metrics_with_prob,
    )

    tester.evaluate()

def pretrain_gat(batch_size=32):

    train_path = r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/train.pkl'
    valid_path = r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/valid.pkl'
    test_path = r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/test.pkl'
    dataset_path = r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset/text_classi/dataset.pkl'

    dataset = pd.read_pickle(dataset_path)
    train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    # Extract text
    train_text = train_data['Content'].tolist()
    valid_text = valid_data['Content'].tolist()
    test_text = test_data['Content'].tolist()
    # Generate embeddings
    train_embeddings = text2vec(train_text, batch_size=batch_size)
    valid_embeddings = text2vec(valid_text, batch_size=batch_size)
    test_embeddings = text2vec(test_text, batch_size=batch_size)
    # Add embeddings to DataFrame
    train_data['mean_pooling_vec'] = train_embeddings.tolist()
    valid_data['mean_pooling_vec'] = valid_embeddings.tolist()
    test_data['mean_pooling_vec'] = test_embeddings.tolist()
    # Save the datasets
    train_data.to_pickle(train_path)
    valid_data.to_pickle(valid_path)
    test_data.to_pickle(test_path)

    print('Text to vector is done!')

    retrieval_data = pd.concat([train_data, valid_data], axis=0)
    retrieval_data.reset_index(drop=True, inplace=True)

    retrieve_vecs = torch.stack([torch.tensor(vec) for vec in retrieval_data['mean_pooling_vec']])

    def retrieval_data(data):
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

    train_data = retrieval(train_data)
    valid_data = retrieval(valid_data)
    test_data = retrieval(test_data)
    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    train_data.to_pickle(train_path)
    valid_data.to_pickle(valid_path)
    test_data.to_pickle(test_path)

    print('Retrieval is done!')

    GAT_main(dataset_path=r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset', 
             dataset_id='text_classi',
                model_id='graph_attention',
                retrieval_num=50,
                epochs=1000,
                batch_size=16,
                early_stop_turns=10,
                loss='BCE',
                optim='Adam',
                lr=1e-5,
                decay_rate=1.0,
                metric='MSE',
                save=r'/home/icdm/CodeSpace/nlp_test/retrieval/result')

    print('GAT is done!')
    return train_path, valid_path, test_path

    

def test_gat():

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
    # print(output)         
    print(predict)
    # print("accuracy:" ,accuracy)
    return output, output_features



# 示例：如何加载保存的模型进行预测
def predict_with_saved_model(text, model_dir):
    loaded_model, loaded_tokenizer = load_model_and_tokenizer(model_dir)
    inputs = loaded_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.item()

def prepare_data_for_prediction():
    # 读取txt文件并解析
    train_file_path = "/home/icdm/CodeSpace/nlp_test/train_data_32k.txt"
    # test_file_path = "/home/icdm/CodeSpace/nlp_test/test_data_32k.txt"
    train_data = parse_txt_file(train_file_path)
    # test_data = parse_txt_file(test_file_path)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # 转换为 Hugging Face Dataset 格式
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    # test_dataset = Dataset.from_list(test_data)

    def preprocess_labels(example):
        example['labels'] = int(example['labels'])
        return example

    train_dataset = train_dataset.map(preprocess_labels)
    val_dataset = val_dataset.map(preprocess_labels)
    # test_dataset = test_dataset.map(preprocess_labels)

    return train_dataset, val_dataset

if __name__ == "__main__":

    train_dataset, val_dataset = prepare_data_for_prediction()

    # pretrain_model(train_dataset, val_dataset)
    # pretrain_gat()
    # test_gat()
    model_test(train_dataset, val_dataset)


    
