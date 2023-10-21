"""
前面的实现多任务的loss权重都是基于手动设置的，这篇代码使用论文：Multi-Task Learning as Multi-Objective Optimization
的方式来生成自适应的权重，获得最优效果。
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from torch.autograd import Variable

sys.path.append('/../image_part')
from layer import *
import warnings
from utils import *
from align_loss import *
from min_norm_solvers import gradient_normalizers, MinNormSolver

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


"""
提取图像特征
"""


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


"""
提取图像加文本加图特征
"""


class EnCoder(nn.Module):
    def __init__(self, config, adj, original_adj):
        super(EnCoder, self).__init__()
        self.config = config
        self.uV = adj.shape[0]
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        maxlen = config['maxlen']

        self.mh_attention = TransformerBlock(input_size=300, n_heads=8, attn_dropout=0)
        self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
                                           _weight=torch.from_numpy(embedding_weights))
        self.cosmatrix = self.calculate_cos_matrix()
        self.gat_relation = Signed_GAT(node_embedding=config['node_embedding'], cosmatrix=self.cosmatrix, nfeat=300,
                                       uV=self.uV, nb_heads=1,
                                       original_adj=original_adj, dropout=0)
        self.image_embedding = resnet50()
        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])
        self.relu = nn.ReLU()

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
        return text_feature, graph_feature, iembedding


# 模态对齐任务
class AlignTask(nn.Module):
    def __init__(self, config):
        super(AlignTask, self).__init__()
        self.config = config
        self.alignfc_g = nn.Linear(in_features=300, out_features=300)
        self.alignfc_t = nn.Linear(in_features=300, out_features=300)
        self.mh_attention = TransformerBlock(input_size=300, n_heads=8, attn_dropout=0)
        self.init_weight()

    def forward(self, text_feature, graph_feature, iembedding):
        bsz = text_feature.size()[0]
        self_att_t = self.mh_attention(text_feature.view(bsz, -1, 300), text_feature.view(bsz, -1, 300), \
                                       text_feature.view(bsz, -1, 300))
        self_att_g = self.mh_attention(graph_feature.view(bsz, -1, 300), graph_feature.view(bsz, -1, 300), \
                                       graph_feature.view(bsz, -1, 300))
        self_att_i = self.mh_attention(iembedding.view(bsz, -1, 300), iembedding.view(bsz, -1, 300), \
                                       iembedding.view(bsz, -1, 300))
        text_enhanced = self.mh_attention(self_att_i.view((bsz, -1, 300)), self_att_t.view((bsz, -1, 300)), \
                                          self_att_t.view((bsz, -1, 300))).view(bsz, 300)
        align_text = self.alignfc_t(text_enhanced)
        align_rembedding = self.alignfc_g(self_att_g)
        dist = [align_text, align_rembedding]
        return dist

    def init_weight(self):
        init.xavier_normal_(self.alignfc_g.weight)
        init.xavier_normal_(self.alignfc_t.weight)


# 分类任务
class ClassfyTask(nn.Module):
    def __init__(self, config):
        super(ClassfyTask, self).__init__()
        self.config = config
        fusion_dim = self.config['fusion_dim']
        self.model_fusion = VLTransformer(fusion_dim, config)
        self.init_weight()

    def forward(self, x):
        prob, _, _ = self.model_fusion(x)
        return prob

    def init_weight(self):
        self.model_fusion.apply(my_weight_init)


# 对比任务
class ConstractiveTask(nn.Module):
    def __init__(self, config):
        super(ConstractiveTask, self).__init__()
        self.config = config
        fusion_dim = self.config['fusion_dim']
        self.model_fusion = VLTransformer(fusion_dim, config)
        self.init_weight()

    def forward(self, x):
        _, hidden, _ = self.model_fusion(x)
        return hidden

    def init_weight(self):
        self.model_fusion.apply(my_weight_init)


class ClassifyContractive(nn.Module):
    def __init__(self, config):
        super(ClassifyContractive, self).__init__()
        self.config = config
        fusion_dim = self.config['fusion_dim']
        self.model_fusion = VLTransformer(fusion_dim, config)
        self.weights = torch.nn.Parameter(torch.ones(2).float())
        self.init_weight()

    def forward(self, x):
        prob, hidden, _ = self.model_fusion(x)
        return prob, hidden

    def init_weight(self):
        self.model_fusion.apply(my_weight_init)


class Model(nn.Module):
    def __init__(self, config, adj, original_adj):
        super(Model, self).__init__()
        self.config = config
        self.encoder = EnCoder(config, adj, original_adj)
        # self.classfy = ClassfyTask(config)
        # self.constractive = ConstractiveTask(config)
        self.decoder = ClassifyContractive(config)
        self.align = AlignTask(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_acc = 0


    def forward(self):
        pass

    """
    先使用共享层获得特征，之后对其进行拷贝，对每一个任务进行梯度累加，对梯度进行归一化，
    对梯度来对loss更新，之后使用FrankWolfeSolver算法根据归一化的梯度计算各个loss的权重
    """

    def fit(self, X_train_tid, X_train, y_train,
            X_dev_tid, X_dev, y_dev):
        if torch.cuda.is_available():
            self.cuda()
        batch_size = self.config['batch_size']
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-3, weight_decay=0)

        X_train_tid = torch.LongTensor(X_train_tid)
        X_train = torch.LongTensor(X_train)
        y_train = torch.LongTensor(y_train)

        # DataLoader
        dataset = TensorDataset(X_train_tid, X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        params = [(name, param) for name, param in self.named_parameters()]
        pgd_word = PGD(self, emb_name='word_embedding', epsilon=6, alpha=1.8)

        # 三个任务对应的损失函数
        loss_align_func = nn.MSELoss()
        loss_cons_func = SupConLoss()
        loss_classify_func = nn.CrossEntropyLoss()

        # Scaling the loss functions based on the algorithm choice
        loss_data = {}
        grads = {}
        scale = {}
        for epoch in range(self.config['epochs']):
            print("\nEpoch ", epoch + 1, "/", self.config['epochs'])
            self.train()
            for idx, data in enumerate(dataloader):
                total = len(dataloader)
                batch_x_tid, batch_x_text, batch_y = (item.cuda(device=self.device) for item in data)
                optimizer.zero_grad()
                # 计算共享层的前向传播
                x_tid = Variable(batch_x_tid, volatile=True)
                x_text = Variable(batch_x_text, volatile=True)
                text_feature, graph_feature, iembedding = self.encoder(x_tid, x_text)

                # 拷贝结果，优化梯度和loss
                text = Variable(text_feature.data.clone(), requires_grad=True)
                graph = Variable(graph_feature.data.clone(), requires_grad=True)
                image = Variable(iembedding.data.clone(), requires_grad=True)

                task_losses = []
                # 特定任务的前向传播
                # 分类任务
                optimizer.zero_grad()
                fusion_input = torch.concat([text, image, graph], dim=-1)
                prob, hidden = self.decoder(fusion_input)
                loss_classify = loss_classify_func(prob, batch_y)
                loss_data["classify"] = loss_classify
                task_losses.append(loss_data["classify"])
                # loss_classify.backward(retain_graph=True)
                # grads["classify"] = []
                # grads["classify"].append(Variable(text.grad.data.clone(), requires_grad=False))

                # 对比任务
                # optimizer.zero_grad()
                features = torch.cat([hidden.unsqueeze(1), hidden.unsqueeze(1)], dim=1)
                loss_cons = loss_cons_func(features, batch_y)
                loss_data["constractive"] = loss_cons
                task_losses.append(loss_data["constractive"])
                # loss_cons.backward()
                # grads["constractive"] = []
                # grads["constractive"].append(Variable(text.grad.data.clone(), requires_grad=False))
                # text.grad.data.zero_()


                task_loss = torch.stack(task_losses)
                weighted_task_loss = torch.mul(self.decoder.weights, task_loss)
                if idx == 0:
                    # set L(0)
                    if torch.cuda.is_available():
                        initial_task_loss = task_loss.data.cpu()
                    else:
                        initial_task_loss = task_loss.data
                    initial_task_loss = initial_task_loss.numpy()

                # get the total loss
                loss = torch.sum(weighted_task_loss)
                # clear the gradients
                optimizer.zero_grad()
                # do the backward pass to compute the gradients for the whole set of weights
                # This is equivalent to compute each \nabla_W L_i(t)
                loss.backward(retain_graph=True)

                self.decoder.weights.grad.data = self.decoder.weights.grad.data * 0.0

                W = self.decoder.model_fusion.Outputlayer
                norms = []
                for i in range(len(task_loss)):
                    # get the gradient of this task loss with respect to the shared parameters
                    gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                    # compute the norm
                    norms.append(torch.norm(torch.mul(self.decoder.weights[i], gygw[0])))
                norms = torch.stack(norms)

                if torch.cuda.is_available():
                    loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
                else:
                    loss_ratio = task_loss.data.numpy() / initial_task_loss
                # r_i(t)
                inverse_train_rate = loss_ratio / np.mean(loss_ratio)

                if torch.cuda.is_available():
                    mean_norm = np.mean(norms.data.cpu().numpy())
                else:
                    mean_norm = np.mean(norms.data.numpy())

                alpha = 0.12
                constant_term = torch.tensor(mean_norm * (inverse_train_rate ** alpha), requires_grad=False)
                if torch.cuda.is_available():
                    constant_term = constant_term.cuda()

                grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                # print('GradNorm loss {}'.format(grad_norm_loss))

                # compute the gradient for the weights
                self.decoder.weights.grad = torch.autograd.grad(grad_norm_loss, self.decoder.weights)[0]
                # # 对齐任务
                # optimizer.zero_grad()
                # dist = self.align(text, graph, image)
                # loss_align = loss_align_func(dist[0], dist[1])
                # loss_data["align"] = loss_align.data
                # task_losses.append(loss_data["align"])
                # loss_align.backward()
                # grads["align"] = []
                # grads["align"].append(Variable(text.grad.data.clone(), requires_grad=False))
                # text.grad.data.zero_()

                # Normalize all gradients, this is optional and not included in the paper.
                # 首先使用梯度的l2范数来对loss加权，再将其用于对梯度进行归一化
                # gn = gradient_normalizers(grads, loss_data, self.config['normalization_type'])
                # tasks = self.config['tasks']
                # for t in tasks:
                #     for gr_i in range(len(grads[t])):
                #         grads[t][gr_i] = grads[t][gr_i] / gn[t]
                #
                # # Frank-Wolfe iteration to compute scales.使用Frank-Wolfe算法来解决约束优化问题，这里得到的就是各个子任务loss的参数
                # sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
                # for i, t in enumerate(tasks):
                #     scale[t] = float(sol[i])
                #
                # # Scaled back-propagation，这里开始对共享层进行反向传播
                # optimizer.zero_grad()
                # text_feature_, graph_feature_, iembedding_ = self.encoder(batch_x_tid, batch_x_text)
                # loss = 0
                # fusion_input_ = torch.concat([text_feature_, graph_feature_, iembedding_], dim=-1)
                # prob_, hidden_ = self.decoder(fusion_input_)
                # loss_cls = loss_classify_func(prob_, batch_y)
                # loss += scale["classify"] * loss_cls
                #
                # features_ = torch.cat([hidden_.unsqueeze(1), hidden_.unsqueeze(1)], dim=1)
                # loss_cons_ = loss_cons_func(features_, batch_y)
                # loss += scale["constractive"] * loss_cons_
                #
                # dist = self.align(text_feature_, graph_feature_, iembedding_)
                # loss_align_ = loss_align_func(dist[0], dist[1])
                # loss += scale["align"] * loss_align_
                #
                # loss.backward()
                # optimizer.step()

                # 对抗性训练
                # K = 3
                # pgd_word.backup_grad()
                # for t in range(K):
                #     pgd_word.attack(is_first_attack=(t == 0))
                #     if t != K - 1:
                #         self.zero_grad()
                #     else:
                #         pgd_word.restore_grad()
                #     text_, graph_, image_ = self.encoder(batch_x_tid, batch_x_text)
                #     fusion_input_prob = torch.concat([text_, image_, graph_], dim=-1)
                #     prob_pgd, _ = self.decoder(fusion_input_prob)
                #     loss_adv = loss_classify_func(prob_pgd, batch_y)
                #     loss_adv.backward()
                # pgd_word.restore()

                optimizer.step()
                corrects = (torch.max(prob, 1)[1].view(batch_y.size()).data == batch_y.data).sum()
                accuracy = 100 * corrects / len(batch_y)
                print(
                    'Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(idx + 1, total,
                                                                                   loss.item(),
                                                                                   accuracy,
                                                                                   corrects,
                                                                                   batch_y.size(0)))
            # 开始验证
            self.evaluate(X_dev_tid, X_dev, y_dev)

    def evaluate(self, X_dev_tid, X_dev, y_dev):
        y_pred = self.predict(X_dev_tid, X_dev)
        acc = accuracy_score(y_dev, y_pred)

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y_dev, y_pred, target_names=self.config['target_names'], digits=5))
            print("Val set acc:", acc)
            print("Best val set acc:", self.best_acc)
            print("saved model at ", self.config['save_path'])

    def predict(self, X_test_tid, X_test):
        if torch.cuda.is_available():
            self.cuda()
        self.eval()
        y_pred = []
        X_test_tid = torch.LongTensor(X_test_tid).cuda()
        X_test = torch.LongTensor(X_test).cuda()

        dataset = TensorDataset(X_test_tid, X_test)
        dataloader = DataLoader(dataset, batch_size=50)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_tid, batch_x_text = (item.cuda(device=self.device) for item in data)
                text, graph, image = self.encoder(batch_x_tid, batch_x_text)
                fusion_input = torch.concat([text, graph, image], dim=-1)
                prob, _ = self.decoder(fusion_input)
                predicted = torch.max(prob, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred


def train_and_test(model):
    model.fit(X_train_tid, X_train, y_train,
              X_dev_tid, X_dev, y_dev)

    model.load_state_dict(torch.load(
        "./exp_result/pheme/exp_description/best_model_in_each_config/Thread-1_configsingle3_best_model_weights_mfan"))
    y_pred = model.predict(X_test_tid, X_test)
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


if __name__ == '__main__':
    config = process_config(config_file.config)
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    model_suffix = "mfan"
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
        0] + '_best_model_weights_' + model_suffix)
    dir_path = os.path.join('exp_result', args.task, args.description)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(config['save_path']):
        os.system('rm {}'.format(config['save_path']))

    X_train_tid, X_train, y_train, \
    X_dev_tid, X_dev, y_dev, \
    X_test_tid, X_test, y_test, adj = load_dataset()
    original_adj = load_original_adj(adj)
    model = Model(config, adj, original_adj)
    train_and_test(model)
