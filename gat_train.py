import logging
import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from retrieval.dataset import MyData, custom_collate_fn
from retrieval.model.GCN_module import GCN as my_model
import random
import numpy as np

from scipy.stats import spearmanr

# 文本颜色设置

BLUE = '\033[94m'
ENDC = '\033[0m'


def seed_init(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_init_msg(logger, args):

    # 打印训练的一些参数
    logger.info(BLUE + 'Random Seed: ' + ENDC + f"{args.seed} ")
    logger.info(BLUE + 'Device: ' + ENDC + f"{args.device} ")
    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_id} ")
    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")
    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")
    logger.info(BLUE + "Optimizer: " + ENDC + f"{args.optim}(lr = {args.lr})")
    # logger.info(BLUE + "Learning Decay: " + ENDC + f"{args.decay_rate}")
    logger.info(BLUE + "Total Epoch: " + ENDC + f"{args.epochs} Turns")
    logger.info(BLUE + "Retrieval Num: " + ENDC + f"{args.retrieval_num}")
    logger.info(BLUE + "Early Stop: " + ENDC + f"{args.early_stop_turns} Turns")
    logger.info(BLUE + "Batch Size: " + ENDC + f"{args.batch_size}")
    logger.info(BLUE + "Training Starts!" + ENDC)

def make_saving_folder_and_logger(args):
    # 创建文件夹和日志文件，用于记录训练结果和模型
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 定义文件夹名称
    folder_name = f"train_{args.model_id}_{args.dataset_id}_{args.retrieval_num}_{args.metric}_{timestamp}"
    # 指定文件夹的完整路径
    father_folder_name = args.save
    if not os.path.exists(father_folder_name):
        os.makedirs(father_folder_name)
    folder_path = os.path.join(father_folder_name, folder_name)
    # 创建文件夹
    os.mkdir(folder_path)
    os.mkdir(os.path.join(folder_path, "trained_model"))
    # 配置日志记录
    # 创建logger对象
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 创建文件处理器
    file_handler = logging.FileHandler(f'{father_folder_name}/{folder_name}/log.txt')
    file_handler.setLevel(logging.INFO)
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # 将处理器添加到logger对象中
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return father_folder_name, folder_name, logger

def delete_model(father_folder_name, folder_name, min_turn):
    model_name_list = os.listdir(f"{father_folder_name}/{folder_name}/trained_model")
    for i in range(len(model_name_list)):
        if model_name_list[i] != f'model_{min_turn}.pth':
            os.remove(os.path.join(f'{father_folder_name}/{folder_name}/trained_model', model_name_list[i]))

def force_stop(msg):
    print(msg)
    sys.exit(1)

def train_val(args):
    # 通过args解析出所有参数
    father_folder_name, folder_name, logger = make_saving_folder_and_logger(args)
    # device
    device = torch.device(args.device)
    # 加载数据集
    train_data = MyData(args.retrieval_num, os.path.join(args.dataset_path, args.dataset_id, 'train.pkl'))
    valid_data = MyData(args.retrieval_num, os.path.join(os.path.join(args.dataset_path, args.dataset_id, 'valid.pkl')))
    train_data_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    valid_data_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    # 加载模型
    model = my_model(args.retrieval_num)
    model = model.to(device)

    if args.loss == 'BCE':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.loss == 'PR':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.loss == 'MSE':
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError('无效的损失函数参数！')

    loss_fn.to(device)
    if args.optim == 'Adam':
        optim = Adam(model.parameters(), args.lr)
    elif args.optim == 'SGD':
        optim = SGD(model.parameters(), args.lr)
    else:
        force_stop('Invalid parameter optim!')

    # 定义学习率衰减
    decayRate = args.decay_rate
    # 定义训练过程的一些参数
    min_total_valid_loss = 1008611
    min_turn = 0

    # 开始训练
    print_init_msg(logger, args)
    for i in range(args.epochs):
        logger.info(f"-----------------------------------Epoch {i + 1} Start!-----------------------------------")
        min_train_loss, total_valid_loss, accuracy = run_one_epoch(args.retrieval_num, model, loss_fn, optim, train_data_loader, valid_data_loader,
                                                        device)
        # scheduler.step()
        logger.info(f"[ Epoch {i + 1} (train) ]: loss = {min_train_loss}")
        logger.info(f"[ Epoch {i + 1} (valid) ]: total_loss = {total_valid_loss}")

        if total_valid_loss < min_total_valid_loss:
            min_total_valid_loss = total_valid_loss
            min_turn = i + 1
        logger.critical(
            f"Current Best Total Loss comes from Epoch {min_turn} , min_total_loss = {min_total_valid_loss}, current_best_accuracy = {accuracy}")
        torch.save(model, f"{father_folder_name}/{folder_name}/trained_model/model_{i + 1}.pth")
        logger.info("Model has been saved successfully!")
        if (i + 1) - min_turn > args.early_stop_turns:
            break
    # delete_model(father_folder_name, folder_name, min_turn)
    logger.info(BLUE + "Training is ended!" + ENDC)

def run_one_epoch(retrieval_num, model, loss_fn, optim, train_data_loader, valid_data_loader, device):
    # 训练部分
    model.train()
    min_train_loss = 1008611
    for batch in tqdm(train_data_loader, desc='Training Progress'):

        batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
        A, mean_pooling_vec, retrieved_label_list, retrieved_textual_feature_embedding, label = batch
        target = label.type(torch.float32)
        output, _ = model.forward(A, mean_pooling_vec, retrieved_label_list, retrieved_textual_feature_embedding)
        # loss = loss_fn(output, target)
        loss_cn = torch.nn.CrossEntropyLoss()
        loss = loss_cn(output.view(-1,2), target.squeeze(-1).type(torch.long))
        optim.zero_grad()
        # 通过损失，优化参数
        loss.backward()
        optim.step()
        if min_train_loss > loss:
            min_train_loss = loss

    # 验证环节
    model.eval()
    total_valid_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_data_loader, desc='Validating Progress'):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

            A, mean_pooling_vec, retrieved_label_list, retrieved_textual_feature_embedding, label = batch
            target = label.type(torch.float32)
            output, _ = model.forward(A, mean_pooling_vec, retrieved_label_list, retrieved_textual_feature_embedding)
            loss_cn = torch.nn.CrossEntropyLoss()
            # loss = loss_fn(output, target)
            loss = loss_cn(output.view(-1,2), target.squeeze(-1).type(torch.long))
            accuracy = torch.sum(torch.argmax(output, dim=-1) == target.squeeze(-1)) / len(target)
            total_valid_loss += loss

    return min_train_loss, total_valid_loss, accuracy


# 主函数，所有训练参数在这里调整
def main():
    # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 运行前命令行参数设置
    parser.add_argument('--seed', default='2024', type=str, help='value of random seed')
    parser.add_argument('--device', default='cuda:0', type=str, help='device used in training')
    parser.add_argument('--metric', default='MSE', type=str, help='the judgement of the training')
    parser.add_argument('--save', default=r'/home/icdm/CodeSpace/nlp_test/retrieval/result', type=str, help='folder to save the results')
    parser.add_argument('--epochs', default=1000, type=int, help='max number of training epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
    parser.add_argument('--early_stop_turns', default=10, type=int, help='early stop turns of training')
    parser.add_argument('--loss', default='BCE', type=str, help='loss function, options: BCE, MSE')
    parser.add_argument('--optim', default='Adam', type=str, help='optim, options: SGD, Adam')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--decay_rate', default=1.0, type=float, help='learning rate decay rate')
    parser.add_argument('--dataset_id', default='text_classi', type=str, help='id of dataset')
    parser.add_argument('--dataset_path', default=r'/home/icdm/CodeSpace/nlp_test/retrieval/dataset', type=str, help='path of dataset')
    parser.add_argument('--model_id', default='graph_attention', type=str, help='id of model')
    parser.add_argument('--retrieval_num', default=50, type=int, help='number of retrieval')
    args = parser.parse_args()

    # 设置随机种子
    seed_init(args.seed)
    train_val(args)


if __name__ == '__main__':

    main()
