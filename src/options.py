#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # arguments related to conformal prediction
    parser.add_argument('--tdigest_delta', type=float, default=0.01)
    parser.add_argument('--tdigest_k', type=int, default=25)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--num_malicious_clients', type=int, default=10)
    # parser.add_argument('--nearest_neighbor_k', type=int, default=10)
    parser.add_argument('--vec_dim', type=int, default=20)
    parser.add_argument('--attack_type', type=str, choices=['coverage','efficiency','gaussian_noise','copy_attack'], default='coverage')
    parser.add_argument('--gaussian_noise_scale', type=float, default=0.5)
    parser.add_argument('--robust_conformal', type=int, default=0)
    parser.add_argument('--scores2vector', type=str, choices=['histogram', 'GMM', 'his2', 'kmeans'], default='histogram')
    parser.add_argument('--adaptive_calibration', type=int, default=0)
    parser.add_argument('--num_round', type=int, default=2)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--num_mal_est_round', type=int, default=5)

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--plot_interval', type=int, default=2)
    parser.add_argument('--plot_path', type=str, default='./test.png')
    parser.add_argument('--num_est_only', type=int, default=0)
    args = parser.parse_args()
    return args
