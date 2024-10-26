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
import ast
from transformers.trainer_utils import EvalPrediction

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


# 预处理标签
def preprocess_labels(example):
    example['labels'] = int(example['labels'])
    return example


train_dataset = train_dataset.map(preprocess_labels)
val_dataset = val_dataset.map(preprocess_labels)
# test_dataset = test_dataset.map(preprocess_labels)

# 加载模型和分词器
model_name = "/home/icdm/CodeSpace/nlp_test/roberta-base/"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)


# 对content部分做tokenizer
def tokenize_function(examples):
    return tokenizer(examples['Content'], padding="max_length", truncation=True)


train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
# test_dataset = test_dataset.map(tokenize_function, batched=True)

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
            "Pred": prob.argmax(-1),
            "Confidence": prob.max().item()  # 预测的概率
        }
        eval_results.append(result)

    # 将eval_results列表转换为DataFrame并保存或输出
    eval_results_df = pd.DataFrame(eval_results)
    eval_results_df.to_csv("evaluation_results.csv", index=False, encoding='utf-8')
    print("Evaluation results saved to evaluation_results.csv")

    # 返回 accuracy 以便 Trainer 继续正常的评估流程
    return {"accuracy": accuracy}

# def compute_metrics(pred):
#     predictions, labels = pred
#     predictions = torch.tensor(predictions)
#     preds = torch.argmax(predictions, dim=1)
#     labels = torch.tensor(labels)
#     accuracy = (preds == labels).float().mean().item()
#     return {"accuracy": accuracy}
#

# 初始化 Trainer

# 训练模型
# trainer.train()

model_dir = "/home/icdm/CodeSpace/nlp_test/pretrained_best"
loaded_model, loaded_tokenizer = load_model_and_tokenizer(model_dir)
# 打印最佳模型的评估结果
trainer = Trainer(
    model=loaded_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=loaded_tokenizer,
    compute_metrics=compute_metrics_with_prob,  # 使用新的计算指标函数
)

best_eval_results = trainer.evaluate()
print(f"Best Validation results: {best_eval_results}")

# 保存最佳模型
# final_output_dir = "/home/icdm/CodeSpace/nlp_test/pretrained_best"
# save_model_and_tokenizer(trainer.model, tokenizer, final_output_dir)


# 示例：如何加载保存的模型进行预测
def predict_with_saved_model(text, model_dir):
    loaded_model, loaded_tokenizer = load_model_and_tokenizer(model_dir)
    inputs = loaded_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.item()




# 使用示例
if __name__ == "__main__":
    sample_text = "这是一个测试文本"
    model_dir = "/home/icdm/CodeSpace/nlp_test/pretrained_best"

    try:
        prediction = predict_with_saved_model(sample_text, model_dir)
        print(f"预测结果：{prediction}")
    except Exception as e:
        print(f"预测过程中出现错误：{str(e)}")
