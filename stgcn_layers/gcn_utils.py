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

    def __init__(self, layout='custom', strategy='uniform', max_hop=1, dilation=1, pose_format="rmtpose_2d"):
        self.max_hop = max_hop
        self.dilation = dilation

        if pose_format == "rtmpose_2d":
            self._get_edge_rtm(layout)
        elif "mediapipe" in pose_format:
            self._get_edge_mp(layout)
        else:
            raise ValueError("Unsupported pose format. Use 'openpose' or 'mediapipe'.")
    
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def _get_edge_mp(self, layout):
        """
        Build the skeleton edges for one of the supported layouts.
        The ‘face_all’ branch below has been *down-sampled*: every other
        point in the inner face chain (0-24) has been removed.
        """
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
        
        # ───────────────────────────────────────────────────────────────
        # BODY  ▸ keep nodes 0, 7-22  → 17 nodes
        # ───────────────────────────────────────────────────────────────
        elif layout == 'body':
            # indices that survive
            keep = [0, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16]
            old2new = {old: new for new, old in enumerate(keep)}
            self.num_node = len(keep)               # 17
            self.center   = old2new[0]              # still node 0 → new 0

            # original neighbour list (old indices)
            neighbour_old = [
                (0, 11), (0, 12), (11, 13), (13, 15),
                (12, 14), (14, 16), (0, 7),  (0, 8)
            ]

            # drop edges touching discarded nodes & remap surviving ones
            neighbour_link = [
                (old2new[a], old2new[b]) for a, b in neighbour_old
                if a in old2new and b in old2new
            ]

            self_link = [(i, i) for i in range(self.num_node)]
            self.edge      = self_link + neighbour_link

        # ───────────────────────────────────────────────────────────────
        #  Face (reduced): keep 0,2,4,…,24  ▸ 13 nodes
        #                  keep outer ring  ▸  8 nodes
        #                  synthetic centre ▸  1 node
        #                  -------------------------------
        #                                    22 nodes total
        # ───────────────────────────────────────────────────────────────
        elif layout == 'face_all':
            # every 3rd point in the original face chain (0–24 range)
            base_idx = [
                # reduced inner chain (7 points)
                56, 38, 46, 47, 108, 100, 117,
                # full outer ring (8 points)
                23, 25, 3, 87, 85, 91, 4, 61,
                # synthetic center
                128
            ]
            old2new = {old: new for new, old in enumerate(base_idx)}
            self.num_node = len(old2new)  # 16
            self.center = old2new[128]

            # self edges
            self.self_link = [(i, i) for i in range(self.num_node)]

            # inner chain edges (7 nodes → 6 edges)
            inner = list(range(7))
            self.neighbor_link = [(inner[i], inner[i + 1]) for i in range(len(inner) - 1)]

            # outer ring edges (8 → 8 edges, wrapped)
            outer = list(range(7, 15))
            self.neighbor_link += [(outer[i], outer[(i + 1) % len(outer)]) for i in range(len(outer))]

            # # connect everything to center
            # self.neighbor_link += [(self.center, i) for i in range(self.num_node) if i != self.center]

            self.edge = self.self_link + self.neighbor_link
        
        return self.edge, self.center

    def _get_edge_rtm(self, layout):
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