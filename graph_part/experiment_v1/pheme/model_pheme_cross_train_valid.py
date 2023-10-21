"""
多模态融合+对比学习
其中对比学习用于将同标签的距离拉近，不同标签的距离拉远
使用对比学习进行模态对齐
使用交叉验证解决过拟合问题
"""

import os

from ClippyAdam import ClippyAdam

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.init as init
import pickle
import json, os, time
import argparse
import config_file
import random
import sys
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

sys.path.append('/../image_part')
from layer import *
import warnings
from utils import *
from align_loss import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.description = "ini"
parser.add_argument("-t", "--task", type=str, default="pheme")
parser.add_argument("-g", "--gpu_id", type=str, default="1")
parser.add_argument("-c", "--config_name", type=str, default="single3.json")
parser.add_argument("-T", "--thread_name", type=str, default="Thread-1")
parser.add_argument("-d", "--description", type=str, default="exp_description")
args = parser.parse_args()


def process_config(config):
    for k, v in config.items():
        config[k] = v[0]
    return config


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.init_clip_max_norm = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler()

    @abc.abstractmethod
    def forward(self):
        pass

    def mfan(self, x_tid, x_text, y, loss, i, total, params, pgd_word, fold=None, fold_idx=None):
        self.optimizer.zero_grad()
        # 这里调用mfan的forward
        logit_original, dist_og, hidden = self.forward(x_tid, x_text)
        loss_classification = loss(logit_original, y)
        # 模态对齐损失
        loss_mse = nn.MSELoss()
        loss_dis = loss_mse(dist_og[0], dist_og[1])
        # loss_mse = ContrastiveLoss()
        # loss_dis = loss_mse.contrastive_loss(dist_og[0], dist_og[1].squeeze(1),
        #                                      batch_size=self.config['batch_size'])

        # 有监督对比损失,使得同标签距离更近，不同标签距离更远
        loss_cons = SupConLoss()
        features = torch.concat([hidden.unsqueeze(1), hidden.unsqueeze(1)], dim=1)
        loss_constrative = loss_cons(features=features, labels=y)

        losses = [loss_classification, loss_dis, loss_constrative]

        important = [loss_classification]
        # print(f"此时的分类损失为：{loss_classification},模态对齐损失为{loss_dis},对比损失为{loss_constrative}")
        loss_defense = geometric_loss(losses, important)
        # loss_defense=0.76*loss_classification+0.12*loss_dis+0.12*loss_constrative
        loss_defense.backward()

        # 对抗性训练
        K = 3
        pgd_word.backup_grad()
        for t in range(K):
            pgd_word.attack(is_first_attack=(t == 0))
            if t != K - 1:
                self.zero_grad()
            else:
                pgd_word.restore_grad()
            loss_adv, dist, _ = self.forward(x_tid, x_text)
            loss_adv = loss(loss_adv, y)
            loss_adv.backward()
        pgd_word.restore()
        # nn.utils.clip_grad_norm_(self.parameters(), max_norm=20, norm_type=2)  # 使用第二种裁剪方式
        self.optimizer.step()
        corrects = (torch.max(logit_original, 1)[1].view(y.size()).data == y.data).sum()
        accuracy = 100 * corrects / len(y)
        if fold is None:
            print(
                'Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                               loss_defense.item(),
                                                                               accuracy,
                                                                               corrects,
                                                                               y.size(0)))
        else:
            print(
                'Fold[{}/{}]   Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(fold_idx + 1, fold,
                                                                                             i + 1, total,
                                                                                             loss_defense.item(),
                                                                                             accuracy,
                                                                                             corrects,
                                                                                             y.size(0)))
        return loss_defense, accuracy / 100

    def fit(self, X_train_tid, X_train, y_train,
            X_dev_tid, X_dev, y_dev, fold, fold_idx):

        if torch.cuda.is_available():
            self.cuda()
        batch_size = self.config['batch_size']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-3, weight_decay=0.0005)
        # self.optimizer = ClippyAdam(self.parameters(), lr=2e-3)

        X_train_tid = torch.LongTensor(X_train_tid)
        X_train = torch.LongTensor(X_train)
        y_train = torch.LongTensor(y_train)

        # DataLoader
        dataset = TensorDataset(X_train_tid, X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss = nn.CrossEntropyLoss()
        params = [(name, param) for name, param in self.named_parameters()]
        pgd_word = PGD(self, emb_name='word_embedding', epsilon=6, alpha=1.8)
        train_losses, train_accs = [], []
        for epoch in range(self.config['epochs']):
            print("\nEpoch ", epoch + 1, "/", self.config['epochs'])
            self.train()
            for i, data in enumerate(dataloader):
                total = len(dataloader)
                batch_x_tid, batch_x_text, batch_y = (item.cuda(device=self.device) for item in data)
                train_loss, train_acc = self.mfan(batch_x_tid, batch_x_text, batch_y, loss, i, total, params, pgd_word,
                                                  fold, fold_idx)
                train_losses.append(train_loss.data.cpu().numpy())
                train_accs.append(train_acc.data.cpu().numpy())
                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)
            valid_loss, acc = self.evaluate(X_dev_tid, X_dev, y_dev, fold_idx)
        train_l = np.mean(train_losses)
        train_a = np.mean(train_accs)
        return train_l, valid_loss, train_a, acc

    def evaluate(self, X_dev_tid, X_dev, y_dev, fold_idx=None):
        y_pred, logits = self.predict(X_dev_tid, X_dev)
        acc = accuracy_score(y_dev, y_pred)
        loss_fun = nn.CrossEntropyLoss()
        valid_loss = loss_fun(logits, y_dev)
        if acc > self.best_acc:
            self.best_acc = acc
            # torch.save(self.state_dict(), self.config['save_path'])
            if fold_idx is not None:
                self.config[
                    'save_path'] = f'./exp_result/pheme/exp_description/best_model_in_each_config/Thread-1_configsingle3_best_model_weights_mfan_train_valid_{fold_idx}'
            else:
                self.config[
                    'save_path'] = './exp_result/pheme/exp_description/best_model_in_each_config/Thread-1_configsingle3_best_model_weights_mfan_train_valid'
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y_dev, y_pred, target_names=self.config['target_names'], digits=5))
            print("Val set acc:", acc)
            print("Best val set acc:", self.best_acc)
            print("saved model at ", self.config['save_path'])
        return valid_loss, acc

    def predict(self, X_test_tid, X_test):
        if torch.cuda.is_available():
            self.cuda()
        self.eval()
        y_pred = []
        X_test_tid = torch.LongTensor(X_test_tid).cuda()
        X_test = torch.LongTensor(X_test).cuda()

        dataset = TensorDataset(X_test_tid, X_test)
        dataloader = DataLoader(dataset, batch_size=50)
        logitses = []
        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_tid, batch_x_text = (item.cuda(device=self.device) for item in data)
                logits, _, _ = self.forward(batch_x_tid, batch_x_text)
                logitses += logits.data.cpu().numpy().tolist()
                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred, torch.tensor(logitses)


class resnet50():
    def __init__(self):
        self.newid2imgnum = config['newid2imgnum']
        self.model = models.resnet50(pretrained=True).cuda()
        self.model.fc = nn.Linear(2048, 300).cuda()
        torch.nn.init.eye_(self.model.fc.weight)
        self.path = os.path.dirname(os.getcwd()) + '/dataset/pheme/pheme_image/pheme_images_jpg/'
        self.trans = self.img_trans()

    def img_trans(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        return transform

    def forward(self, xtid):
        img_path = []
        img_list = []
        for newid in xtid.cpu().numpy():
            imgnum = self.newid2imgnum[newid]
            imgpath = self.path + imgnum + '.jpg'
            im = np.array(self.trans(Image.open(imgpath)))
            im = torch.from_numpy(np.expand_dims(im, axis=0)).to(torch.float32)
            img_list.append(im)
        batch_img = torch.cat(img_list, dim=0).cuda()
        img_output = self.model(batch_img)
        return img_output


class MFAN(NeuralNetwork):
    def __init__(self, config, adj, original_adj):
        super(MFAN, self).__init__()
        self.config = config
        self.uV = adj.shape[0]
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        self.mh_attention = TransformerBlock(input_size=300, n_heads=8, attn_dropout=0)
        self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
                                           _weight=torch.from_numpy(embedding_weights))
        self.cosmatrix = self.calculate_cos_matrix()
        self.gat_relation = Signed_GAT(node_embedding=config['node_embedding'], cosmatrix=self.cosmatrix, nfeat=300, \
                                       uV=self.uV, nb_heads=1,
                                       original_adj=original_adj, dropout=0)
        self.image_embedding = resnet50()
        fusion_dim = config['fusion_dim']
        self.model_fusion = VLTransformer(fusion_dim, config)
        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.fc1 = nn.Linear(in_features=32, out_features=self.config['num_classes'])
        self.fc2 = nn.Linear(in_features=105, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=256, out_features=105)
        self.alignfc_g = nn.Linear(in_features=300, out_features=300)
        self.alignfc_t = nn.Linear(in_features=300, out_features=300)
        self.init_weight()

    def calculate_cos_matrix(self):
        a, b = torch.from_numpy(config['node_embedding']), torch.from_numpy(config['node_embedding'].T)
        c = torch.mm(a, b)
        aa = torch.mul(a, a)
        bb = torch.mul(b, b)
        asum = torch.sqrt(torch.sum(aa, dim=1, keepdim=True))
        bsum = torch.sqrt(torch.sum(bb, dim=0, keepdim=True))
        norm = torch.mm(asum, bsum)
        res = torch.div(c, norm)
        return res

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)
        self.model_fusion.apply(my_weight_init)

    def forward(self, X_tid, X_text):
        # 获取文本嵌入
        X_text = self.word_embedding(X_text)
        if self.config['user_self_attention'] == True:
            X_text = self.mh_attention(X_text, X_text, X_text)
        X_text = X_text.permute(0, 2, 1)
        # 从GAT获取传播结构
        rembedding = self.gat_relation.forward(X_tid)
        # resnet50获取图像嵌入
        iembedding = self.image_embedding.forward(X_tid)
        conv_block = [rembedding]
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(X_text))
            pool = max_pooling(act)
            pool = torch.squeeze(pool, dim=-1)
            conv_block.append(pool)

        conv_feature = torch.cat(conv_block, dim=1)
        graph_feature, text_feature = conv_feature[:, :300], conv_feature[:, 300:]
        bsz = text_feature.size()[0]

        # text_feature->[bach_size,300],graph_feature->[bach_size,300],iembedding->[batch_size,300]
        fusion_input = torch.concat([text_feature, iembedding, graph_feature], dim=-1)
        prob, hidden, attn = self.model_fusion(fusion_input)

        self_att_t = self.mh_attention(text_feature.view(bsz, -1, 300), text_feature.view(bsz, -1, 300), \
                                       text_feature.view(bsz, -1, 300))
        self_att_g = self.mh_attention(graph_feature.view(bsz, -1, 300), graph_feature.view(bsz, -1, 300), \
                                       graph_feature.view(bsz, -1, 300))
        self_att_i = self.mh_attention(iembedding.view(bsz, -1, 300), iembedding.view(bsz, -1, 300), \
                                       iembedding.view(bsz, -1, 300))
        text_enhanced = self.mh_attention(self_att_i.view((bsz, -1, 300)), self_att_t.view((bsz, -1, 300)), \
                                          self_att_t.view((bsz, -1, 300))).view(bsz, 300)

        # 特征对齐
        align_text = self.alignfc_t(text_enhanced)
        align_rembedding = self.alignfc_g(self_att_g)
        dist = [align_text, align_rembedding]

        # MLP
        output = self.relu(self.bn1(self.fc2(hidden)))
        output = self.relu(self.bn2(self.fc3(output)))
        output = self.dropout(output)
        output = self.fc1(output)
        output = F.sigmoid(output)

        return output, dist, hidden


def load_dataset():
    pre = os.path.dirname(os.getcwd()) + '/dataset/pheme/pheme_files/'
    X_train_tid, X_train, y_train, word_embeddings, adj = pickle.load(open(pre + "/train.pkl", 'rb'))
    X_dev_tid, X_dev, y_dev = pickle.load(open(pre + "/dev.pkl", 'rb'))
    X_test_tid, X_test, y_test = pickle.load(open(pre + "/test.pkl", 'rb'))
    config['embedding_weights'] = word_embeddings
    config['node_embedding'] = pickle.load(open(pre + "/node_embedding.pkl", 'rb'))[0]
    print("#nodes: ", adj.shape[0])

    with open(pre + '/new_id_dic.json', 'r') as f:
        newid2mid = json.load(f)
        newid2mid = dict(zip(newid2mid.values(), newid2mid.keys()))
    content_path = os.path.dirname(os.getcwd()) + '/dataset/pheme/'
    with open(content_path + '/content.csv', 'r') as f:
        reader = csv.reader(f)
        result = list(reader)[1:]
        mid2num = {}
        for line in result:
            mid2num[line[1]] = line[0]
    newid2num = {}
    for id in X_train_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_dev_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_test_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    config['newid2imgnum'] = newid2num

    return X_train_tid, X_train, y_train, \
           X_dev_tid, X_dev, y_dev, \
           X_test_tid, X_test, y_test, adj


def load_original_adj(adj):
    pre = os.path.dirname(os.getcwd()) + '/dataset/pheme/pheme_files/'
    path = os.path.join(pre, 'original_adj')
    with open(path, 'r') as f:
        original_adj_dict = json.load(f)
    original_adj = np.zeros(shape=adj.shape)
    for i, v in original_adj_dict.items():
        v = [int(e) for e in v]
        original_adj[int(i), v] = 1
    return original_adj


def train_and_test():
    model_suffix = 'mfan'
    res_dir = '../../exp_result'
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
        0] + '_best_model_weights_' + model_suffix + '_train_valid')
    dir_path = os.path.join('exp_result', args.task, args.description)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(config['save_path']):
        os.system('rm {}'.format(config['save_path']))

    X_train_tid, X_train, y_train, \
    X_dev_tid, X_dev, y_dev, \
    X_test_tid, X_test, y_test, adj = load_dataset()
    original_adj = load_original_adj(adj)

    # nn.fit(X_train_tid, X_train, y_train,
    #        X_dev_tid, X_dev, y_dev)

    train_k_fold(5, X_train_tid, X_train, y_train,
                 X_dev_tid, X_dev, y_dev, config, adj, original_adj)

    # 开始测试
    # 取出输出每一个fold的结果
    for i in range(5):
        print('*' * 25, f'第{i + 1}个fold的测试结果如下：', '*' * 25)
        nn = MFAN(config, adj, original_adj)
        nn.load_state_dict(torch.load(
            f"./exp_result/pheme/exp_description/best_model_in_each_config/Thread-1_configsingle3_best_model_weights_mfan_train_valid_{i}"))
        y_pred, _ = nn.predict(X_test_tid, X_test)
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


# 共k折，取第i折的数据
def get_k_fold_data(k, i, X_tid, X, y):
    # idx=[i for i in range(len(X_train_tid))]
    fold_size = X.shape[0] // k
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid_tid, X_valid, y_valid = X_tid[val_start:val_end], X[val_start:val_end], y[val_start:val_end]
        X_train_tid = torch.cat([X_tid[0:val_start], X_tid[val_end:]], dim=0)
        X_train = torch.cat([X[0:val_start], X[val_end:]], dim=0)
        y_train = torch.cat([y[0:val_start], y[val_end:]], dim=0)
    else:  # 若是最后一折交叉验证
        X_valid_tid, X_valid, y_valid = X_tid[val_start:], X[val_start:], y[val_start:]  # 若不能整除，将多的case放在最后一折里
        X_train_tid = X_tid[0:val_start]
        X_train = X[0:val_start]
        y_train = y[0:val_start]

    return X_train_tid, X_train, y_train, X_valid_tid, X_valid, y_valid


# K折交叉验证
def train_k_fold(k, X_train_tid, X_train, y_train,
                 X_dev_tid, X_dev, y_dev, config, adj, original_adj):
    X_train_tid = torch.LongTensor(X_train_tid)
    X_train = torch.LongTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_dev_tid = torch.LongTensor(X_dev_tid)
    X_dev = torch.LongTensor(X_dev)
    y_dev = torch.LongTensor(y_dev)

    X_tid = torch.cat([X_train_tid, X_dev_tid], dim=0)
    X = torch.cat([X_train, X_dev], dim=0)
    y = torch.cat([y_train, y_dev], dim=0)

    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    for fold in range(k):
        print('*' * 25, '第', fold + 1, '折', '*' * 25)
        model = MFAN(config, adj, original_adj)
        X_train_tid, X_train, y_train, X_valid_tid, X_valid, y_valid = get_k_fold_data(k, fold, X_tid, X, y)
        train_loss, val_loss, train_acc, val_acc = model.fit(X_train_tid, X_train, y_train, X_valid_tid,
                                                             X_valid, y_valid, k, fold)
        print('train_loss:{:.5f}, train_acc:{:.3f}%'.format(train_loss, train_acc * 100))
        print('valid loss:{:.5f}, valid_acc:{:.3f}%\n'.format(val_loss, val_acc * 100))

        train_loss_sum += train_loss
        valid_loss_sum += val_loss
        train_acc_sum += train_acc
        valid_acc_sum += val_acc

    print('\n', '#' * 10, f'最终{k}折交叉验证结果', '#' * 10)

    print('average train loss:{:.4f}, average train accuracy:{:.3f}%'.format(train_loss_sum / k,
                                                                             train_acc_sum * 100 / k))
    print('average valid loss:{:.4f}, average valid accuracy:{:.3f}%'.format(valid_loss_sum / k,
                                                                             valid_acc_sum * 100 / k))


config = process_config(config_file.config)
seed = config['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
train_and_test()
