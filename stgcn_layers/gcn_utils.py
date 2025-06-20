import torch
import numpy as np
import torch.nn as nn
import pdb
import math
import copy


class Graph:
    """The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self, layout='custom', strategy='uniform', max_hop=1, dilation=1, pose_format="openpose"):
        self.max_hop = max_hop
        self.dilation = dilation

        if pose_format == "openpose":
            self.get_edge(layout)
        elif pose_format == "mediapipe":
            self._get_edge_mp(layout)
        else:
            raise ValueError("Unsupported pose format. Use 'openpose' or 'mediapipe'.")
    
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        # 'body', 'left', 'right', 'mouth', 'face'
        # if layout == 'custom_hand21':
        if layout == 'left' or layout == 'right':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [0, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [0, 9],
                [9, 10],
                [10, 11],
                [11, 12],
                [0, 13],
                [13, 14],
                [14, 15],
                [15, 16],
                [0, 17],
                [17, 18],
                [18, 19],
                [19, 20],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
        
        elif layout == 'body':
            self.num_node = 9
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [3, 5],
                [5, 7],
                [4, 6],
                [6, 8],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'face_all':
            self.num_node = 9 + 8 + 1
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [[i, i + 1] for i in range(9 - 1)] + \
                             [[i, i + 1] for i in range(9, 9 + 8 - 1)] + \
                             [[9 + 8 - 1, 9]] + \
                             [[17, i] for i in range(17)]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = self.num_node - 1
    
    def _get_edge_mp(self, layout):
        if layout == 'left' or layout == 'right':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [0, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [0, 9],
                [9, 10],
                [10, 11],
                [11, 12],
                [0, 13],
                [13, 14],
                [14, 15],
                [15, 16],
                [0, 17],
                [17, 18],
                [18, 19],
                [19, 20],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
        
        elif layout == 'body':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 11],
                [0, 12],
                [11, 13],
                [13, 15],
                [12, 14],
                [14, 16],
                [0, 7],
                [0, 8],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0

        elif layout == 'face_all':
            self.num_node = 129
            self_link = [] #[(i, i) for i in range(num_node)]
            neighbor_1base = [
                # FACE OVAL
                [56, 37],
                [37, 65],
                [65, 31],
                [31, 38],
                [38, 16],
                [16, 58],
                [58, 40],
                [40, 46],
                [46, 45],
                [45, 60],
                [60, 44],
                [44, 47],
                [47, 106],
                [106, 121],
                [121, 107],
                [107, 108],
                [108, 102],
                [102, 119],
                [119, 78],
                [78, 100],
                [100, 93],
                [93, 126],
                [126, 99],
                [99, 117],
                # MOUTH
                [23, 25],
                [25, 3],
                [3, 87],
                [87, 85],
                [85, 91],
                [91, 4],
                [4, 61],
                [61, 23],
            ] + [
                # MAP ALL POINTS TO BODY[0]
                (128, i) for i in
                [3, 4, 16, 23, 25, 31, 37, 38, 40, 44, 45, 46, 47, 56, 58, 60, 61, 65, 78, 85, 87, 91, 93, 99, 100, 102, 106, 107, 108, 117, 119, 121, 126]
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = self.num_node - 1


    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if (
                                self.hop_dis[j, self.center]
                                == self.hop_dis[i, self.center]
                            ):
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif (
                                self.hop_dis[j, self.center]
                                > self.hop_dis[i, self.center]
                            ):
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    print("Computing Hop Distance Matrix..., num_node:", num_node, "max_hop:", max_hop)
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD