import warnings

warnings.filterwarnings("ignore")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import csv
import abc
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pickle
import json, os
import argparse
import sys
import ClippyAdam
import align_loss
import config_file
import random
from utils import *
from PIL import Image
from layer import *
import datetime
import copy

sys.path.append('/../image_part')

parser = argparse.ArgumentParser()
parser.description = "ini"
parser.add_argument("-t", "--task", type=str, default="pheme")
parser.add_argument("-g", "--gpu_id", type=str, default="1")
parser.add_argument("-c", "--config_name", type=str, default="single3.json")
parser.add_argument("-T", "--thread_name", type=str, default="Thread-1")
parser.add_argument("-d", "--description", type=str, default="exp_description")
args = parser.parse_args()


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.init_clip_max_norm = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abc.abstractmethod
    def forward(self):
        pass

    def mfan(self, x_tid, x_text, y, loss, i, total, params, pgd_word):
        self.optimizer.zero_grad()
        # 这里调用mfan的forward
        (logit_0, logit_1), dist_og = self.forward(x_tid, x_text)
        # print(logit_original)
        # print(y)

        # 分类损失
        loss_classification_0 = loss(logit_0, y)
        # loss_classification_1 = loss(logit_1, y)
        loss_classification = loss_classification_0

        # # R-drop
        # kl_loss = align_loss.kl_loss(logit_0, logit_1)

        # 模态对齐
        loss_mse = nn.MSELoss()
        loss_dis = loss_mse(dist_og[0], dist_og[1].squeeze(1))

        # loss_align = align_loss.MultimodalContrastiveLoss(device=self.device, batch_size=dist_og[0].shape[0], temp=0.05)
        # # 模态对齐损失
        # loss_dis_1 = align_loss.constractive_loss(dist_og[0], dist_og[1].squeeze(1))  # 文本与传播图对齐
        # loss_dis_2 = align_loss.constractive_loss(dist_og[0], dist_og[2].squeeze(1))  # 文本与图像对齐
        # loss_dis = 1.0 * (loss_dis_1 + loss_dis_2) / 2

        # loss_defense = 0.4 * loss_classification + 0.3 * kl_loss + 0.3 * loss_dis
        loss_defense = 1.8 * loss_classification + 2.4 * loss_dis
        # loss_defense = loss_classification
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
            (logit_adv_0, logit_adv_1), dist = self.forward(x_tid, x_text)
            loss_adv_0 = loss(logit_adv_0, y)
            loss_adv_1 = loss(logit_adv_1, y)
            loss_adv = (loss_adv_0 + loss_adv_1) * 0.5
            loss_adv.backward()
        pgd_word.restore()
        self.optimizer.step()
        corrects = (torch.max((logit_adv_0 + logit_adv_1) / 2, 1)[1].view(y.size()).data == y.data).sum()
        accuracy = 100 * corrects / len(y)
        print(
            'Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                           loss_defense.item(),
                                                                           accuracy,
                                                                           corrects,
                                                                           y.size(0)))
        return loss_defense.item(), accuracy

    def fit(self, X_train_tid, X_train, y_train,
            X_dev_tid, X_dev, y_dev):

        if torch.cuda.is_available():
            self.cuda()
        batch_size = self.config['batch_size']
        # self.optimizer = ClippyAdam.ClippyAdam(self.parameters(), lr=2e-3)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)

        X_train_tid = torch.LongTensor(X_train_tid)
        X_train = torch.LongTensor(X_train)
        y_train = torch.LongTensor(y_train)

        # DataLoader
        dataset = TensorDataset(X_train_tid, X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss = nn.CrossEntropyLoss()
        params = [(name, param) for name, param in self.named_parameters()]
        pgd_word = PGD(self, emb_name='word_embedding', epsilon=6, alpha=1.8)
        avg_loss_list = []
        avg_acc_list = []
        valid_acc_list = []
        valid_loss_list = []
        early_stopping = EarlyStopping()
        for epoch in range(self.config['epochs']):
            print("\nEpoch ", epoch + 1, "/", self.config['epochs'])
            self.train()
            loss_list = []
            acc_list = []
            for i, data in enumerate(dataloader):
                total = len(dataloader)
                batch_x_tid, batch_x_text, batch_y = (item.cuda(device=self.device) for item in data)
                loss_, acc_ = self.mfan(batch_x_tid, batch_x_text, batch_y, loss, i, total, params, pgd_word)
                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)
                loss_list.append(loss_)
                acc_list.append(acc_.cpu())
            valid_acc, valid_loss = self.evaluate(X_dev_tid, X_dev, y_dev, loss)
            valid_acc_list.append(np.mean(valid_acc))
            valid_loss_list.append(torch.mean(valid_loss, dim=-1))
            avg_acc_list.append(np.mean(acc_list))
            avg_loss_list.append(np.mean(loss_list))
            early_stopping(valid_loss, model)
            # 达到早停止条件时，early_stop会被置为True
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break  # 跳出迭代，结束训练
        return avg_loss_list, avg_acc_list, valid_acc_list, valid_loss_list, epoch

    def evaluate(self, X_dev_tid, X_dev, y_dev, loss_func):
        correct, logits = self.predict(X_dev_tid, X_dev)
        acc = accuracy_score(y_dev, correct)
        # print(torch.Tensor(y_dev))
        # print(torch.Tensor(logits))
        loss = loss_func(torch.tensor(logits, dtype=torch.float), torch.tensor(y_dev, dtype=torch.long))
        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y_dev, correct, target_names=self.config['target_names'], digits=5))
            print("Val set acc:", acc)
            print("Best val set acc:", self.best_acc)
            print("saved model at ", self.config['save_path'])
        return acc.item() * 100, loss

    def predict(self, X_test_tid, X_test):
        if torch.cuda.is_available():
            self.cuda()
        self.eval()
        y_pred = []
        logits_list = []
        X_test_tid = torch.LongTensor(X_test_tid).cuda()
        X_test = torch.LongTensor(X_test).cuda()

        dataset = TensorDataset(X_test_tid, X_test)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_tid, batch_x_text = (item.cuda(device=self.device) for item in data)
                (logits_0, logits_1), dist = self.forward(batch_x_tid, batch_x_text)
                logits = (logits_0 + logits_1) / 2
                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
                logits_list += logits.data.cpu().numpy().tolist()
        return y_pred, logits_list


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
        self.gat_relation = Signed_GAT(node_embedding=config['node_embedding'], cosmatrix=self.cosmatrix, nfeat=300,
                                       uV=self.uV, nb_heads=1,
                                       original_adj=original_adj, dropout=0)
        self.image_embedding = resnet50()
        # 多模态融合模块
        fusion_dim = config['fusion_dim']
        self.model_fusion = VLTransformer(fusion_dim, config)

        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(105, 32)
        self.layer_norm_1 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(in_features=32, out_features=config['num_classes'])
        self.alignfc_g = nn.Linear(in_features=300, out_features=300)
        self.alignfc_t = nn.Linear(in_features=300, out_features=300)
        self.alignfc_i = nn.Linear(in_features=300, out_features=300)
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
        init.xavier_normal_(self.alignfc_g.weight)
        init.xavier_normal_(self.alignfc_t.weight)
        init.xavier_normal_(self.alignfc_i.weight)
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

        fusion_input2 = torch.concat([text_feature, graph_feature, iembedding], dim=-1)
        prob_1, _, _ = self.model_fusion(fusion_input2)
        bsz = text_feature.size()[0]

        self_att_t = self.mh_attention(text_feature.view(bsz, -1, 300), text_feature.view(bsz, -1, 300),
                                       text_feature.view(bsz, -1, 300))
        self_att_g = self.mh_attention(graph_feature.view(bsz, -1, 300), graph_feature.view(bsz, -1, 300),
                                       graph_feature.view(bsz, -1, 300))
        self_att_i = self.mh_attention(iembedding.view(bsz, -1, 300), iembedding.view(bsz, -1, 300),
                                       iembedding.view(bsz, -1, 300))
        text_enhanced = self.mh_attention(self_att_i.view((bsz, -1, 300)), self_att_t.view((bsz, -1, 300)),
                                          self_att_t.view((bsz, -1, 300))).view(bsz, 300)

        # 特征对齐
        align_text = self.alignfc_t(text_enhanced.squeeze(1))
        align_graph = self.alignfc_g(self_att_g.squeeze(1))
        align_image = self.alignfc_i(self_att_i.squeeze(1))
        dist = [align_text, align_graph]

        # text_feature->[bach_size,300],graph_feature->[bach_size,300],iembedding->[batch_size,300]
        fusion_input = torch.concat([align_text, align_graph, align_image], dim=-1)

        prob_0, _, _ = self.model_fusion(fusion_input)

        return (prob_0, prob_1), dist


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


def train_and_test(model):
    model_suffix = model.__name__.lower().strip("text")
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
    if os.path.exists(config['save_path']):
        os.system('rm {}'.format(config['save_path']))

    X_train_tid, X_train, y_train, \
    X_dev_tid, X_dev, y_dev, \
    X_test_tid, X_test, y_test, adj = load_dataset()
    original_adj = load_original_adj(adj)
    nn = model(config, adj, original_adj)

    loss_list, acc_list, valid_acc_list, valid_loss_list, epoch_num = nn.fit(X_train_tid, X_train, y_train,
                                                                             X_dev_tid, X_dev, y_dev)
    x = [i + 1 for i in range(epoch_num + 1)]
    x_label = "Epoch"
    y1_label = "Acc"
    y2_label = "Loss"
    time = datetime.datetime.now()
    time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    draw_picture(x, loss_list, x_label, y2_label, x2=x, y2=valid_loss_list, title=f"Loss Pheme(Train) {time_str}")
    draw_picture(x, acc_list, x_label, y1_label, x2=x, y2=valid_acc_list, title=f"Acc Pheme(Train) {time_str}")

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


if __name__ == '__main__':
    config = process_config(config_file.config)
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = MFAN
    train_and_test(model)
