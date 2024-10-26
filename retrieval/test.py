import argparse
import os
from datetime import datetime
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from dataset import MyData, custom_collate_fn
import random
import numpy as np

BLUE = '\033[94m'
ENDC = '\033[0m'

def seed_init(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_init_msg(logger, args):
    logger.info(BLUE + 'Random Seed: ' + ENDC + f"{args.seed} ")
    logger.info(BLUE + 'Device: ' + ENDC + f"{args.device} ")
    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_path} ")
    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")
    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")
    logger.info(BLUE + "Testing Starts!" + ENDC)

def delete_special_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    content = content.replace(BLUE, '')
    content = content.replace(ENDC, '')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def test(args):
    device = torch.device(args.device)
    model_id = args.model_id
    dataset_id = args.dataset_id
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"test_{model_id}_{dataset_id}__{args.retrieval_num}_{timestamp}"
    father_folder_name = args.save

    if not os.path.exists(father_folder_name):
        os.makedirs(father_folder_name)

    folder_path = os.path.join(father_folder_name, folder_name)
    os.mkdir(folder_path)

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{father_folder_name}/{folder_name}/log.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    test_data = MyData(args.retrieval_num, os.path.join(os.path.join(args.dataset_path, args.dataset_id, 'test.pkl')))
    test_data_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    model = torch.load(args.model_path)
    total_test_step = 0
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    total_acc = 0

    print_init_msg(logger, args)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc='Testing'):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            A, mean_pooling_vec, retrieved_label_list, retrieved_textual_feature_embedding, label = batch
            label = label.type(torch.float32)
            output, output_features = model.forward(A, mean_pooling_vec, retrieved_label_list, retrieved_textual_feature_embedding)

            # output = output.to('cpu').numpy()
            # label = label.to('cpu').numpy()
            # output = np.where(output > 0.5, 1, 0)

            # total_recall += recall_score(label, output, average='macro')
            # total_precision += precision_score(label, output, average='macro')
            total_acc += torch.sum(torch.argmax(output, dim=-1) == label.squeeze(-1)) / len(label)
            # total_f1 += f1_score(label, output, average='macro')
            # total_acc += accuracy_score(label, output)

            total_test_step += 1

    logger.warning(f"[ Test Result ]:  \n Accuracy = {total_acc / total_test_step}"
                   f"\nRecall = {total_recall / total_test_step}"
                   f"\nPrecision = {total_precision / total_test_step}"
                   f"\nF1 = {total_f1 / total_test_step}\n")

    logger.info("Test is ended!")
    delete_special_tokens(f"{father_folder_name}/{folder_name}/log.txt")

def main(args):
    seed_init(args.seed)
    test(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default='2024', type=str, help='value of random seed')
    parser.add_argument('--device', default='cuda:0', type=str, help='device used in testing')
    parser.add_argument('--metric', default='ACC, F1, RECALL', type=str, help='the judgement of the testing')
    parser.add_argument('--save', default=r'/home/icdm/CodeSpace/nlp_test/retrieval/test_result', type=str, help='folder to save the results')
    parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
    parser.add_argument('--dataset_id', default='text_classi', type=str, help='id of dataset')
    parser.add_argument('--dataset_path', default=r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset', type=str, help='path of dataset')
    parser.add_argument('--model_id', default='graph_attention', type=str, help='id of model')
    parser.add_argument('--retrieval_num', default=50, type=int, help='number of retrieval')
    parser.add_argument('--model_path',
                        default=r"/home/icdm/CodeSpace/nlp_test/retrieval/result/train_graph_attention_text_classi_50_MSE_2024-10-26_22-45-05/trained_model/model_15.pth",
                        type=str, help='path of trained model')

    args = parser.parse_args()
    main(args)