import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import os
import copy
import time
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
from tdigest import TDigest
import random
import heapq
from sklearn import mixture
from scipy.stats import norm
from tdigest_improve import Distributed_TDigest
import pyudorandom
from likelihood_est_num_mal import estimate_likelihood
from likelihood_est_num_mal import estimate_likelihood_v2
# import warnings
# warnings.filterwarnings('ignore')
import torchvision.models as models
import torch.nn as nn
from resnet import ResNet18
from certification import certification

def test(global_model, test_loader):
    global_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc

def cal_conform_scores(output, gt):
    num = gt.shape[0]
    res = torch.zeros((num)).cuda()
    for j in range(num):
        res[j] = output[j,gt[j]].exp()
    return res

def score2index(score, unit):
    return int(score/unit)-1

def score2vec(scores, vec_dim, scores2vector):
    if scores2vector=='histogram':
        histogram = np.zeros(vec_dim)
        unit = 1.0 / vec_dim
        for score in scores:
            ind = score2index(score, unit)
            histogram[ind] += 1.0
        histogram = histogram / len(scores)
        return histogram
    elif scores2vector=='GMM':
        mu, std = norm.fit(scores)
        vec = np.array([mu, std])
        return vec
    elif scores2vector=='his2':
        vec = np.zeros(vec_dim)
        for i in range(1,vec_dim):
            cur = np.quantile(scores, 1.0*i/vec_dim)
            vec[i] = cur
        return vec



def cal_distance(vec1,vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def cal_score(distances, k):
    ind = heapq.nlargest(k, range(len(distances)), distances.__getitem__)
    return distances[ind].sum()

def detect_mal(scores_list, num_mal, nearest_neighbor_k, vec_dim, scores2vector, return_score=False):
    score_vector = []
    for score in scores_list:
        score_vector.append(score2vec(score, vec_dim, scores2vector))
    num = len(scores_list)
    distance_mat = np.zeros((num,num))
    scores = np.zeros(num)
    for i in range(num):
        for j in range(num):
            distance_mat[i][j] = cal_distance(score_vector[i], score_vector[j])
    for i in range(num):
        scores[i] = cal_score(distance_mat[i], k=nearest_neighbor_k)
    if return_score:
        return score_vector, scores, heapq.nlargest(num_mal, range(len(scores)), scores.__getitem__)
    return heapq.nlargest(num_mal, range(len(scores)), scores.__getitem__)

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('../..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    # exp_details(args)

    # if args.gpu:
    #     torch.cuda.set_device('cuda:'+args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    if args.dataset in ['mnist','cifar']:
        K=10
    elif args.dataset in ['tinyimagenet']:
        K=200

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    elif args.model == 'resnet':
        if args.dataset == 'cifar':
            global_model = ResNet18(num_classes=10)
        elif args.dataset == 'tinyimagenet':
            global_model = models.resnet18()
            global_model.avgpool = nn.AdaptiveAvgPool2d(1)
            global_model.fc.out_features = 200
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    if args.dataset != 'tinyimagenet':
        global_model = torch.load(
            './save/global_model_{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.model, args.epochs, args.frac,
                                                              args.iid, args.local_ep, args.local_bs))
    else:
        if args.iid == 1:
            global_model.load_state_dict(torch.load(f'./save/global_model_tinyimagenet_resnet_iid.pt'))
        elif args.iid == 0:
            global_model.load_state_dict(torch.load(f'./save/global_model_tinyimagenet_resnet_noniid.pt'))
        else:
            exit('args.iid must be in {0,1}')
    global_model.to(device)

    num_instances = len(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    if args.dataset=='tinyimagenet':
        # fix the mismatch of labels in TinyImagenet val set
        small_labels = {}
        with open(os.path.join('./data/tiny-imagenet-200', "words.txt"), "r") as dictionary_file:
            line = dictionary_file.readline()
            while line:
                label_id, label = line.strip().split("\t")
                small_labels[label_id] = label
                line = dictionary_file.readline()

        labels = {}
        label_ids = {}
        for label_index, label_id in enumerate(train_loader.dataset.classes):
            label = small_labels[label_id]
            labels[label_index] = label
            label_ids[label_id] = label_index

        val_label_map = {}
        with open(os.path.join('./data/tiny-imagenet-200', "val/val_annotations.txt"), "r") as val_label_file:
            line = val_label_file.readline()
            while line:
                file_name, label_id, _, _, _, _ = line.strip().split("\t")
                val_label_map[file_name] = label_id
                line = val_label_file.readline()
        for i in range(len(test_dataset.imgs)):
            file_path = test_dataset.imgs[i][0]

            file_name = os.path.basename(file_path)
            label_id = val_label_map[file_name]

            test_dataset.imgs[i] = (file_path, label_ids[label_id])

    cali_set, test_set = torch.utils.data.random_split(test_dataset, [num_instances//2, num_instances-num_instances//2])

    test_loader = DataLoader(test_set, batch_size=128,shuffle=True)

    test_loss, acc = test(global_model, test_loader)
    # print(f'Test accuracy of the global model: {acc}')

    # conformal prediction
    if args.iid==1:
        cali_data_split = torch.utils.data.random_split(cali_set, [int(len(cali_set) / args.num_users) for _ in range(args.num_users)])
        cali_loader = [torch.utils.data.DataLoader(x, batch_size=128, shuffle=True) for x in cali_data_split]
    elif args.iid==0:
        dataset = cali_set
        min_size = 0
        min_require_size = 10
        N = len(dataset)
        net_dataidx_map = {}
        beta = args.beta
        labels = []
        for i in range(N):
            labels.append(dataset[i][1])
        labels = np.array(labels)
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(args.num_users)]
            for k in range(K):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, args.num_users))
                proportions = np.array([p * (len(idx_j) < N / args.num_users) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(args.num_users):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
        cali_data_split = net_dataidx_map
        cali_loader = [torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,net_dataidx_map[j]), batch_size=128, shuffle=True) for j in range(args.num_users)]


    num_scores_list = []
    for i in range(args.num_users):
        num_scores_list.append(len(cali_data_split[i]))

    scores_list = []
    client_list = [i for i in range(args.num_users)]
    malicious_client_list = random.sample(client_list, args.num_malicious_clients)


    for i in range(args.num_users):
        cur_scores = []
        for batch_idx, (data, target) in enumerate(cali_loader[i]):
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            if args.dataset=='tinyimagenet':
                output = F.log_softmax(output, dim=1)[:,:200]
            cur_scores.append(cal_conform_scores(output,target))
        cur_scores = torch.cat(cur_scores,dim=0)
        scores_list.append(cur_scores.cpu().detach().numpy())

    copy_target: int = 0
    while copy_target in malicious_client_list:
        copy_target = np.random.choice(list(range(args.num_users)),1)[0]

    for i in malicious_client_list:
        if args.attack_type=='coverage':
            scores_list[i] = np.array([1.0] * len(scores_list[i]))
        elif args.attack_type=='efficiency':
            scores_list[i] = np.array([0.0] * len(scores_list[i]))
        elif args.attack_type=='gaussian_noise':
            scores_list[i] = scores_list[i] + np.random.normal(loc=0, scale=args.gaussian_noise_scale, size=scores_list[i].size)
            scores_list[i][scores_list[i] > 1.0] = 1.0
            scores_list[i][scores_list[i] < 0.0] = 0.0
        elif args.attack_type=='copy_attack':
            scores_list[i] = scores_list[copy_target]

    print(f'True number of malicious clients: {args.num_malicious_clients}')
    # estimate number of malicious clients
    num_mal = args.num_users // 2
    for i in range(args.num_mal_est_round):
        scores_vec, scores, benign_ind = detect_mal(scores_list, num_mal=num_mal, nearest_neighbor_k=args.num_users - num_mal - 1,
                                vec_dim=args.vec_dim, scores2vector=args.scores2vector, return_score=True)
        score_index = np.argsort(scores)
        scores_vec = np.array(scores_vec)
        best_liklihood = -1e5
        for j in range(2,args.num_users//2):
            # liklihood = estimate_likelihood(scores_vec, score_index, j)
            liklihood = estimate_likelihood_v2(scores_vec, score_index, j)
            if liklihood > best_liklihood:
                best_liklihood = liklihood
                num_mal = j
        # print(f'Round {i} estimated number of malicious clients: {num_mal}')
    print(f'Estimated number of malicious clients: {num_mal}')
    args.num_malicious_clients = num_mal

    if args.num_est_only==1:
        exit()

    tdigest_delta = args.tdigest_delta
    tdigest_k = args.tdigest_k
    alpha = args.alpha
    digest = TDigest(delta=tdigest_delta, K=tdigest_k)

    if args.robust_conformal==1:
        nearest_neighbor_k = args.num_users - args.num_malicious_clients - 1
        list_mal_clients_detected = detect_mal(scores_list, num_mal=args.num_malicious_clients, nearest_neighbor_k=nearest_neighbor_k, vec_dim=args.vec_dim, scores2vector=args.scores2vector)
    else:
        list_mal_clients_detected = []

    # print(f'Malicious client list: {np.sort(np.array(malicious_client_list))}')
    # print(f'Detected malicious client list: {np.sort(np.array(list_mal_clients_detected))}')

    communication_cost = []
    score_list_full = []

    if args.adaptive_calibration==0:
        N = 0
        for i in range(args.num_users):
            if i not in list_mal_clients_detected:
                N += len(scores_list[i])
                client_digest = TDigest(delta=tdigest_delta, K=tdigest_k)
                client_digest.batch_update(np.array(scores_list[i]))
                score_list_full = score_list_full + list(scores_list[i])
                communication_cost.append(len(client_digest)*2) # for each item, we have the mean value and corresponding weight
                digest = digest + client_digest
        q_hat = digest.percentile(round(100 * (1-np.ceil((N+1)*(1-alpha))/N)))
    elif args.adaptive_calibration==1: # discard this option currently
        delta_list = [0.04, 0.02]
        q_hat = 0.0
        for r in range(args.num_round):
            digest = Distributed_TDigest(delta=delta_list[r])
            for i in range(args.num_users):
                if i not in list_mal_clients_detected:
                    digest_tmp = Distributed_TDigest(delta=delta_list[r])
                    scores = np.array(scores_list[i])
                    if r==0:
                        score_list_full = score_list_full + list(scores_list[i])
                    digest_tmp.target_quantile = np.count_nonzero(scores < q_hat) / scores.size
                    digest_tmp.batch_update(np.array(scores))
                    communication_cost.append(len(digest_tmp)*2)
                    digest = digest + digest_tmp
            q_hat = digest.percentile(round(100 * alpha))
    # print(f'q_hat: {q_hat}')
    # print(f'alpha: {alpha}')
    # print(score_list_full)
    # print(q_hat)
    # print(f'estimated quantile: {np.quantile(np.array(score_list_full), q_hat)}')
    approx_error = abs(q_hat - np.quantile(np.array(score_list_full), alpha))

    communication_cost = np.sum(np.array(communication_cost))
    num_covered = 0
    num_setsize = 0
    num_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            if args.dataset=='tinyimagenet':
                output = F.log_softmax(output, dim=1)[:,:200]
            output_ = torch.exp(output)
            num_setsize += torch.sum(output_>q_hat)
            res = cal_conform_scores(output, target)
            num_covered += torch.sum(res>q_hat)
            num_total += len(target)
    print('Under attack {} with num_malicious_clients {}, communication cost and approximation error of the quantile: {} / {:4f}, avg_coverage and avg_set_size: {:4f} / {:4f}'.format(args.attack_type, args.num_malicious_clients, communication_cost, approx_error, 1.0 * num_covered / num_total, 1.0 * num_setsize/num_total))
    # print(f'avg_set_size: {1.0 * num_setsize/num_total}')