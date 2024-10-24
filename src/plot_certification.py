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

import torchvision.models as models
import torch.nn as nn
from resnet import ResNet18
from certification import certification
from plot_bnd import make_plot

from tqdm import tqdm

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
        return score_vector
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
    # if args.model == 'cnn':
    #     # Convolutional neural netork
    #     if args.dataset == 'mnist':
    #         global_model = CNNMnist(args=args)
    #     elif args.dataset == 'fmnist':
    #         global_model = CNNFashion_Mnist(args=args)
    #     elif args.dataset == 'cifar':
    #         global_model = CNNCifar(args=args)
    #
    # elif args.model == 'mlp':
    #     # Multi-layer preceptron
    #     img_size = train_dataset[0][0].shape
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #         global_model = MLP(dim_in=len_in, dim_hidden=64,
    #                            dim_out=args.num_classes)
    # else:
    #     exit('Error: unrecognized model')
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
            # Finetune Final few layers to adjust for tiny imagenet input
            global_model.avgpool = nn.AdaptiveAvgPool2d(1)
            num_ftrs = global_model.fc.in_features
            global_model.fc = nn.Linear(num_ftrs, 200)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model = torch.load('./save/global_model_{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs))
    global_model.to(device)

    num_instances = len(test_dataset)
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

    ratio = []
    upp_bnd = []
    low_bnd = []
    cov = []
    ratios = []

    scores_list = []
    for i in range(args.num_users):
        cur_scores = []
        for batch_idx, (data, target) in enumerate(cali_loader[i]):
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            cur_scores.append(cal_conform_scores(output, target))
        cur_scores = torch.cat(cur_scores, dim=0)
        scores_list.append(cur_scores.cpu().detach().numpy())

    output_all = []
    output_ori = []
    target_all = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            output_ = torch.exp(output)
            output_ori.append(output)
            output_all.append(output_)
            target_all.append(target)
    output_ori = torch.concatenate(output_ori)
    output_all = torch.concatenate(output_all)
    target_all = torch.concatenate(target_all)

    for num_mal in tqdm(range(1,args.num_users//2,args.plot_interval)):
        ratio.append(num_mal / (args.num_users-num_mal))

        client_list = [i for i in range(args.num_users)]
        malicious_client_list = random.sample(client_list, num_mal)
        lower_bnd,upper_bnd = certification(args.num_users, malicious_client_list, cali_data_split, epsilon=0.0, H=args.vec_dim, beta=1e-1, alpha=args.alpha)
        upp_bnd.append(upper_bnd)
        low_bnd.append(lower_bnd)

        copy_target: int = 0
        while copy_target in malicious_client_list:
            copy_target = np.random.choice(list(range(args.num_users)),1)[0]

        print(f'num_mal: {num_mal}; lower_bound: {lower_bnd}, upper_bound: {upper_bnd}')
        if num_mal<30:
            times = 5
        elif num_mal < 40:
            times = 10
        else:
            times=20
        times = 0
        for time in range(times):
            for i in malicious_client_list:
                if args.attack_type=='coverage':
                    scores_list[i] = np.array([1.0] * len(scores_list[i]))
                elif args.attack_type=='efficiency':
                    scores_list[i] = np.array([0.0] * len(scores_list[i]))
                elif args.attack_type=='gaussian_noise':
                    scale = random.uniform(0,1.0)
                    scores_list[i] = scores_list[i] + np.random.normal(loc=0, scale=scale, size=scores_list[i].size)
                    scores_list[i][scores_list[i] > 1.0] = 1.0
                    scores_list[i][scores_list[i] < 0.0] = 0.0
                elif args.attack_type=='copy_attack':
                    scores_list[i] = scores_list[copy_target]

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


            num_setsize = torch.sum(output_all>q_hat)
            res = cal_conform_scores(output_ori, target_all)
            num_covered = torch.sum(res>q_hat)
            num_total = len(target)
            cov.append((num_covered/num_total).cpu().item())
            ratios.append(ratio[-1])

    # print(low_bnd)
    # print(upp_bnd)
    # print(ratio)
    make_plot(ratio,low_bnd,upp_bnd,ratios,cov,path=args.plot_path)

