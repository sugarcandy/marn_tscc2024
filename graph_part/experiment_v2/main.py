import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_loading import load_original_adj_pheme, load_dataset_pheme, load_dataset_weibo, load_original_adj_weibo
import numpy as np
import random
from model import MARN
import config_file as config_file
from align_loss import *
from utils import *
from layer import PGD
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.description = "ini"
parser.add_argument("-t", "--task", type=str, default="pheme")
parser.add_argument("-g", "--gpu_id", type=str, default="1")
parser.add_argument("-c", "--config_name", type=str, default="single3.json")
parser.add_argument("-T", "--thread_name", type=str, default="Thread-1")
parser.add_argument("-d", "--description", type=str, default="exp_description2")
args = parser.parse_args()


def process_config(config):
    for k, v in config.items():
        config[k] = v[0]
    return config


def train_and_test():
    model_suffix = "mfan"
    res_dir = 'exp_result'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.task)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.description)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = config['save_path'] = os.path.join(res_dir, 'best_model_in_each_config')
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    config['save_path'] = os.path.join(res_dir, args.thread_name + '_' + 'config' + args.config_name.split(".")[
        0] + '_best_model_weights_' + model_suffix)
    dir_path = os.path.join('exp_result', args.task, args.description)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # if os.path.exists(config['save_path']):
    #     os.system('rm {}'.format(config['save_path']))

    if args.task == "pheme":
        X_train_tid, X_train, y_train, \
        X_dev_tid, X_dev, y_dev, \
        X_test_tid, X_test, y_test, adj = load_dataset_pheme(config)
        original_adj = load_original_adj_pheme(adj)
    elif args.task == "weibo":
        X_train_tid, X_train, y_train, \
        X_dev_tid, X_dev, y_dev, \
        X_test_tid, X_test, y_test, adj = load_dataset_weibo(config)
        original_adj = load_original_adj_weibo(adj)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MARN(config, adj, original_adj, args)

    model = model.to(device)

    batch_size = config['batch_size']
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=0)
    # self.optimizer = ClippyAdam(self.parameters(), lr=2e-3)

    X_train_tid = torch.LongTensor(X_train_tid)
    X_train = torch.LongTensor(X_train)
    y_train = torch.LongTensor(y_train)

    # DataLoader
    dataset = TensorDataset(X_train_tid, X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss = nn.CrossEntropyLoss()
    params = [(name, param) for name, param in model.named_parameters()]
    pgd_word = PGD(model, emb_name='word_embedding', epsilon=6, alpha=1.8)
    best_acc = 0
    for epoch in range(config['epochs']):
        print("\nEpoch ", epoch + 1, "/", config['epochs'])
        model.train()
        avg_loss = 0
        avg_acc = 0
        for i, data in enumerate(dataloader):
            total = len(dataloader)
            batch_x_tid, batch_x_text, batch_y = (item.cuda() for item in data)
            optimizer.zero_grad()
            # 这里调用mfan的forward
            logit_original, dist_og, hidden, features = model(batch_x_tid, batch_x_text)
            loss_classification = loss(logit_original, batch_y)
            # 分类损失
            loss_mse = nn.MSELoss()
            # 模态对齐损失
            loss_dis = loss_mse(dist_og[0], dist_og[1].squeeze())

            # hidden的shape为[batch_size,output_dim]
            # 此时features的shape为[batch_size,2,output_dim],为每一份数据创建了一个副本，进行对比
            loss_cons = SupConLoss()
            loss_constrative = loss_cons(features, batch_y)

            losses = []
            losses.append(loss_classification)
            losses.append(loss_constrative)
            losses.append(loss_dis)


            important = []
            important.append(loss_classification)
            if args.task=="pheme":
                loss_defense = geometric_loss(losses)
            else:
                loss_defense = geometric_loss(losses,important)
            loss_defense.backward()

            # 对抗性训练
            K = 3
            pgd_word.backup_grad()
            for t in range(K):
                pgd_word.attack(is_first_attack=(t == 0))
                if t != K - 1:
                    optimizer.zero_grad()
                else:
                    pgd_word.restore_grad()
                loss_adv, dist, _, _ = model(batch_x_tid, batch_x_text)
                loss_adv = loss(loss_adv, batch_y)
                loss_adv.backward()
            pgd_word.restore()
            if args.task == "pheme":
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)  # 使用第二种裁剪方式
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)  # 使用第二种裁剪方式
            optimizer.step()
            corrects = (torch.max(logit_original, 1)[1].view(batch_y.size()).data == batch_y.data).sum()
            accuracy = 100 * corrects / len(batch_y)
            print(
                'Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                               loss_defense.item(),
                                                                               accuracy,
                                                                               corrects,
                                                                               batch_y.size(0)))
        ## 验证
        model.eval()
        y_pred = []
        X_dev_tid = torch.LongTensor(X_dev_tid)
        X_dev = torch.LongTensor(X_dev)

        dev_dataset = TensorDataset(X_dev_tid, X_dev)
        dev_dataloader = DataLoader(dev_dataset, batch_size=50)

        for i, data in enumerate(dev_dataloader):
            with torch.no_grad():
                batch_x_tid, batch_x_text = (item.cuda(device=device) for item in data)
                logits, dist, _, _ = model(batch_x_tid, batch_x_text)
                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
        acc = accuracy_score(y_dev, y_pred)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), config['save_path'])
            print(classification_report(y_dev, y_pred, target_names=config['target_names'], digits=5))
            print("Val set acc:", acc)
            print("Best val set acc:", best_acc)
            print("saved model at ", config['save_path'])
    ## 测试
    model.eval()
    model.load_state_dict(torch.load(
        f"../exp_result/{args.task}/exp_description2/best_model_in_each_config/Thread-1_configsingle3_best_model_weights_mfan"))
    y_pred = []
    X_test_tid = torch.LongTensor(X_test_tid)
    X_test = torch.LongTensor(X_test)

    test_dataset = TensorDataset(X_test_tid, X_test)
    test_dataloader = DataLoader(test_dataset, batch_size=50)

    for i, data in enumerate(test_dataloader):
        with torch.no_grad():
            batch_x_tid, batch_x_text = (item.cuda(device=device) for item in data)
            logits, dist, _, _ = model(batch_x_tid, batch_x_text)
            predicted = torch.max(logits, dim=1)[1]
            y_pred += predicted.data.cpu().numpy().tolist()

    res = classification_report(y_test, y_pred, target_names=config['target_names'], digits=3, output_dict=True)
    for k, v in res.items():
        print(k, v)
    print("result:{:.4f}".format(res['accuracy']))
    res2 = {}
    res_final = {}
    res_final.update(res)
    res_final.update(res2)
    print(res)
    return res


## 加载配置
config = process_config(config_file.config)
seed = config['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
print("**********************************************************************************************************")
print("**********************************************************************************************************")
print("**********************************************************************************************************")
print(f"                                              在{args.task}数据集上的训练                                   ")
print("**********************************************************************************************************")
print("**********************************************************************************************************")
print("**********************************************************************************************************")
train_and_test()
