import argparse
import numpy as np
import torch
import sys
sys.path.append("..") 
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from baseline_models.NCN.model import predictor_dict, convdict, GCN, DropEdge
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from baseline_models.NCN.util import PermIterator
import time
# from ogbdataset import loaddataset
from typing import Iterable
from torch_geometric.datasets import Planetoid,HeterophilousGraphDataset,Amazon,AttributedGraphDataset,Coauthor,CitationFull,LINKXDataset,Actor,WikipediaNetwork
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from baseline_models.utils import init_seed, Logger, save_emb, get_logger
from baseline_models.evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from torch_geometric.transforms import RandomLinkSplit

from baseline_models.utils import *
from collections import Counter
import matplotlib.pyplot as plt

#log_print = get_logger('testrun', 'log', get_config_dir())


import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
import networkx as nx
import random
from torch_geometric.utils import to_dense_adj
import pandas as pd

import os
cwd = os.getcwd()

log_print = get_logger('testrun', 'log', get_config_dir())

from torch_geometric.utils import to_networkx
import networkx as nx
from sklearn.cluster import SpectralClustering


def add_label_noise(labels, noise_rate):
    """
    Applies uniform (symmetric) noise to a tensor of labels.

    Args:
    - labels (torch.Tensor): The original 1D tensor of class labels.
    - noise_rate (float): The percentage of labels to flip (e.g., 0.1 for 10%).

    Returns:
    - noisy_labels (torch.Tensor): A new tensor with noise injected.
    """
    if noise_rate == 0:
        return labels

    noisy_labels = labels.clone()
    num_labels = len(labels)
    num_to_flip = int(num_labels * noise_rate)

    # Get the unique classes to know the possible new labels
    classes = torch.unique(labels).tolist()
    num_classes = len(classes)

    if num_classes < 2:
        print("Warning: Cannot flip labels as there is only one class.")
        return labels

    # Randomly select indices to flip
    indices_to_flip = torch.randperm(num_labels)[:num_to_flip]

    for idx in indices_to_flip:
        original_label = noisy_labels[idx].item()
        
        # Create a list of possible new labels (all classes except the original)
        possible_new_labels = [c for c in classes if c != original_label]
        
        # Randomly choose a new label from the possible options
        if possible_new_labels:
            new_label = random.choice(possible_new_labels)
            noisy_labels[idx] = new_label

    return noisy_labels

import random

def add_label_noise_manual(labels, noise_rate):
    """
    Applies uniform (symmetric) noise to a list of labels.

    Args:
    - labels (list): The original 1D list of class labels.
    - noise_rate (float): The percentage of labels to flip (e.g., 0.1 for 10%).

    Returns:
    - noisy_labels (list): A new list with noise injected.
    """
    if noise_rate == 0:
        return labels

    # Create a copy of the list (replaces labels.clone())
    noisy_labels = list(labels)
    
    num_labels = len(labels)
    num_to_flip = int(num_labels * noise_rate)

    # Get the unique classes (replaces torch.unique().tolist())
    classes = list(set(labels))
    num_classes = len(classes)

    if num_classes < 2:
        print("Warning: Cannot flip labels as there is only one class.")
        return labels

    # Randomly select *unique* indices to flip
    # Replaces torch.randperm(num_labels)[:num_to_flip]
    indices_to_flip = random.sample(range(num_labels), num_to_flip)

    for idx in indices_to_flip:
        # Get the original label (no .item() needed)
        original_label = noisy_labels[idx]
        
        # Create a list of possible new labels
        possible_new_labels = [c for c in classes if c != original_label]
        
        # Randomly choose a new label
        if possible_new_labels:
            new_label = random.choice(possible_new_labels)
            noisy_labels[idx] = new_label

    return noisy_labels

def apply_louvain(data):
    G = to_networkx(data, to_undirected=True)
    partition = nx.community.louvain_communities(G, seed=42)
    cluster_labels = torch.zeros(data.num_nodes, dtype=torch.long)
    for cluster_id, nodes in enumerate(partition):
        cluster_labels[list(nodes)] = cluster_id
    # For consistency, also return normalized features like in K-means
    normalized_features = torch.nn.functional.normalize(data.x, p=2, dim=1)
    return cluster_labels, normalized_features


def apply_spectral(data, k):
    G = to_networkx(data, to_undirected=True)
    A = nx.to_numpy_array(G)
    clustering = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=42)
    cluster_labels = clustering.fit_predict(A)
    cluster_labels = torch.tensor(cluster_labels, dtype=torch.long)
    normalized_features = torch.nn.functional.normalize(data.x, p=2, dim=1)
    return cluster_labels, normalized_features



def get_fb_prob(df,split_edge):
    edges= split_edge['train']['edge']
    h={}
    for i in range(len(edges[0])):
        a=df['page_type'].iloc[int(edges[i][0])]
        b=df['page_type'].iloc[int(edges[i][1])]
        if a not in h:
            h[a]={}
        if b not in h:
            h[b]={}
        if a not in h[b]:h[b][a]=0
        if b not in h[a]:h[a][b]=0
        h[a][b]+=1
        h[b][a]+=1
    for i in h:
        s=0
        for j in h[i]:
            s+=h[i][j]
        for j in h[i]:
            h[i][j]/=s
    return h

def get_cluster_labels(split_edge, data, k=10, max_iters=100):
    """
    Function to perform K-means clustering using cosine similarity on graph node features.
    
    Args:
    - train_edge_index (torch.Tensor): Tensor containing the training edges.
    - data (torch_geometric.data.Data): PyG data object containing node features and graph information.
    - k (int): The number of clusters. Default is 10.
    - max_iters (int): Maximum iterations for K-means clustering. Default is 100.
    
    Returns:
    - cluster_labels (torch.Tensor): Tensor containing the cluster labels for each node.
    """
    # Convert training edges into adjacency matrix

    train_edges = split_edge['train']['edge']
    valid_edges = split_edge['valid']['edge']
    edges = torch.cat([train_edges, valid_edges], dim=0)

    #print(edges)

    train_adj = to_dense_adj(edges, max_num_nodes=data.num_nodes).squeeze(0)

    # Function to normalize the adjacency matrix symmetrically
    def normalize_adjacency(adj):
        """
        Symmetrically normalize the adjacency matrix with added self-loops.
        
        Args:
        - adj (torch.Tensor): The adjacency matrix (dense format).
        
        Returns:
        - adj_normalized (torch.Tensor): The normalized adjacency matrix with self-loops.
        """
        # Add self-loops by adding an identity matrix (I) to the adjacency matrix (A)
        adj_with_self_loops = adj + torch.eye(adj.size(0), device=adj.device)

        # Degree matrix
        degree = torch.sum(adj_with_self_loops, dim=1)  # Degree matrix

        # D^-0.5 (inverse square root of the degree matrix)
        degree_inv_sqrt = torch.pow(degree, -0.5)

        # Handle inf values by replacing them with 0 (for isolated nodes with degree 0)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0

        # Symmetric normalization: D^-0.5 * A * D^-0.5
        adj_normalized = adj_with_self_loops * degree_inv_sqrt.view(-1, 1)  # A * D^-0.5
        adj_normalized = adj_normalized * degree_inv_sqrt.view(1, -1)  # D^-0.5 * A * D^-0.5
        
        return adj_normalized


    # Normalize adjacency matrix
    normalized_train_adj = normalize_adjacency(train_adj)

    # Multiply normalized adjacency matrix with features to get aggregated features
    aggregated_features = torch.matmul(normalized_train_adj, data.x)
    #aggregated_features = data.x

    # Normalize aggregated features for better numerical stability
    normalized_features = torch.nn.functional.normalize(aggregated_features, p=2, dim=1)

    # Initialize random cluster centers (pick k random nodes' embeddings)
    np.random.seed(0)
    random_indices = np.random.choice(normalized_features.size(0), size=k, replace=False)
    cluster_centers = normalized_features[random_indices]

    # Function to compute cosine similarity
    def cosine_similarity(x1, centers):
        # Normalize the input vectors and centers for cosine similarity
        x1_norm = torch.nn.functional.normalize(x1, p=2, dim=1)
        centers_norm = torch.nn.functional.normalize(centers, p=2, dim=1)
        
        # Compute cosine similarity
        similarities = torch.matmul(x1_norm, centers_norm.T)
        
        # Convert similarities to distances by inverting them (1 - similarity)
        distances = 1 - similarities
        return distances

    # K-means clustering loop with cosine similarity
    for iteration in range(max_iters):
        # Compute cosine similarity distance between each node and cluster centers
        distances = cosine_similarity(normalized_features, cluster_centers)

        # Assign each node to the cluster with the smallest cosine distance
        cluster_labels = torch.argmin(distances, dim=1)

        # Update cluster centers (recompute as the mean of assigned points)
        new_cluster_centers = torch.zeros_like(cluster_centers)
        for i in range(k):
            nodes_in_cluster = normalized_features[cluster_labels == i]
            if len(nodes_in_cluster) > 0:
                new_cluster_centers[i] = torch.mean(nodes_in_cluster, dim=0)
            else:
                # If no nodes assigned, keep the cluster center unchanged
                new_cluster_centers[i] = cluster_centers[i]

        # # Check for convergence (if centers don't change, stop)
        if torch.allclose(new_cluster_centers, cluster_centers, atol=1e-6):
            print(f"Converged after {iteration+1} iterations")
            break

        cluster_centers = new_cluster_centers

    return cluster_labels,normalized_features

def find_optimal_k(split_edge,data,name):
    ssd = []  # Sum of squared distances
    k_values = [1,2,5,10,15,20]  # Range of k to test

    for k in k_values:
        # Get cluster labels
        cluster_labels,normalized_features = get_cluster_labels(split_edge, data, k=k, max_iters=100)

        # Compute SSD for this k
        #normalized_features = torch.nn.functional.normalize(data.x, p=2, dim=1)
        cluster_centers = torch.stack([
            torch.mean(normalized_features[cluster_labels == i], dim=0)
            for i in range(k)
        ])
        distances = 1 - torch.matmul(normalized_features, cluster_centers.T)
        ssd.append(torch.sum(distances.min(dim=1).values ** 2).item())

    # Plot the elbow graph
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, ssd, marker='o', linestyle='--', color='b')
    plt.title(name)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Sum of Squared Distances (SSD)")
    plt.xticks(k_values)
    plt.grid()
    #plt.show()
    #plt.grid()
    plt.savefig("/home/ankit/research/CGLE_code/CGLE/results/plots/"+name+".png", dpi=300, bbox_inches='tight')



class MyCustomDataset(InMemoryDataset): #for FB Page-Page network
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyCustomDataset, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # List the files that need to be found in the raw directory
        return ['node_features.pt', 'edge_list.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Implement this method to download raw data if not present
        # For example, you might download files from a URL
        pass

    def process(self):
        # Load the raw data
        node_features = torch.load(self.raw_paths[0])  # 'node_features.pt'
        edge_list = torch.load(self.raw_paths[1])      # 'edge_list.pt'

        # Create a Data object
        data = Data(x=node_features, edge_index=edge_list.contiguous())

        # Optionally apply pre-transformations
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_list = [data]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_class_intersection_prob(y, split_edge, args):
    # Check if the uniform_addon flag is set
    if args.uniform_addon:
        # --- LOGIC FOR UNIFORM (UNINFORMATIVE) PROBABILITIES ---
        print("--- MODIFIED: Using uniform (uninformative) probabilities for the addon feature test. ---")
        unique_classes = torch.unique(y).tolist()
        num_classes = len(unique_classes)

        if num_classes == 0:
            return {}
        
        uniform_prob = 1.0 / num_classes
        
        h = {}
        for class_i in unique_classes:
            h[class_i] = {}
            for class_j in unique_classes:
                # Corrected the variable from 'j' to 'class_j'
                h[class_i][class_j] = uniform_prob
        return h
    else:
        # --- ORIGINAL LOGIC FOR REAL PROBABILITIES ---
        class_count = Counter(y.tolist())
        print(f"Class counts: {class_count}")

        train_edges = split_edge['train']['edge']
        valid_edges = split_edge['valid']['edge']
        edges = torch.cat([train_edges, valid_edges], dim=0)

        h = {}

        for i in range(edges.size(1)):
            a = y[edges[0, i]].item()
            b = y[edges[1, i]].item()
            
            if a not in h: h[a] = {}
            if b not in h: h[b] = {}
            if a not in h[b]: h[b][a] = 0
            if b not in h[a]: h[a][b] = 0

            h[a][b] += 1
            h[b][a] += 1

        for i in h:
            s = 0
            for j in h[i]:
                s += h[i][j]
            if s > 0:
                for j in h[i]:
                    h[i][j] /= s
        return h

def data_fetch_prob(y, data_dict, edges):
    """
    Fetches the probability of interaction between class labels for given edges and applies Laplacian smoothing.

    Parameters:
    y: torch.Tensor
        Tensor containing class labels of the nodes.
    data_dict: dict
        Dictionary containing the probabilities of interaction between class labels (output from get_class_intersection_prob function).
    edges: torch.Tensor
        Tensor of shape [2, num_edges], where each column is an edge between two nodes.

    Returns:
    torch.Tensor
        Tensor of shape [num_edges, 2], where each row contains the smoothed probabilities for an edge.
    """
    pompom = []
    n = edges.shape[1]

    #y=torch.argmax(y, dim=1) ##For Facebook

    for i in range(n):
        src = int(edges[0][i])  # Source node index
        des = int(edges[1][i])  # Destination node index
        
        a = y[src].item()  # Class label of source node
        b = y[des].item()  # Class label of destination node
        
        # Fetch the probabilities of class a to class b and class b to class a
        pab = data_dict.get(a, {}).get(b, 0)  # P(a->b)
        pba = data_dict.get(b, {}).get(a, 0)  # P(b->a)
        
        # Create a list of probabilities
        dung = [pab, pba]
        
        # Apply Laplacian smoothing
        k = 1  # Laplacian smoothing parameter
        V = 2  # Number of possible outcomes (pab, pba)
        dung = [(x + k) / (1 + V * k) for x in dung]
        
        # Append smoothed probabilities
        pompom.append(dung)
    
    # Convert the result into a torch tensor
    pompom = torch.tensor(pompom, dtype=torch.float)
    return pompom



def randomsplit(dataset, data_name,k=0):
   
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
   
    ##############
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []
    node_set = set()
    
    dir_path = cwd
    
    for split in ['train', 'test', 'valid']:

        path = dir_path+'/data_splits' + '/{}/{}_pos.txt'.format(data_name, split)

       
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj:
                continue

            if split == 'train': 
                train_pos.append((sub, obj))
                

            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    
    num_nodes = len(node_set)
    print('the number of nodes in ' + data_name + ' is: ', num_nodes)

    for split in ['test', 'valid']:

        path = dir_path+'/data_splits' + '/{}/{}_neg.txt'.format(data_name, split)

      
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            # if sub == obj:
            #     continue
            
            if split == 'valid': 
                valid_neg.append((sub, obj))
               
            if split == 'test': 
                test_neg.append((sub, obj))

    #train_pos=increase_components_to_k(train_pos, k)
    #train_pos=randomly_remove_edges(train_pos, k)
    #train_pos=randomly_remove_percentage_edges(train_pos, k)

    
    train_pos_tensor = torch.tensor(train_pos)

    

    valid_pos = torch.tensor(valid_pos)
    valid_neg =  torch.tensor(valid_neg)

    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]

    split_edge['train']['edge'] = train_pos_tensor
    # data['train_val'] = train_val

    split_edge['valid']['edge']= valid_pos
    split_edge['valid']['edge_neg'] = valid_neg
    split_edge['test']['edge']  = test_pos
    split_edge['test']['edge_neg']  = test_neg

    return split_edge

def loaddataset(name, use_valedges_as_input, load=None,k=0):

    if name=='fb':
        dataset = MyCustomDataset(root="datasets/fb_page")

    elif name in ["ppi",'TWeibo','Flickr','Facebook']:
        dataset=AttributedGraphDataset(root="datasets", name=name)
    elif name in['CS','Physics']:
        dataset=Coauthor(root="datasets",name=name)
    elif name in ["Roman-empire", "Amazon-ratings","Questions","Minesweeper", "Tolokers"]:
        dataset = HeterophilousGraphDataset(root="datasets", name=name.lower())
    elif name=='DBLP':
        dataset=CitationFull(root='datasets',name=name)
    elif name in ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"]:
        dataset=LINKXDataset(root='datasets', name=name)
    elif name=="actor":
        dataset=Actor(root="datasets")
    elif name in ["chameleon", "crocodile", "squirrel"]:
        dataset=WikipediaNetwork(root="datasets",name=name)
    else:

        dataset = Planetoid(root="datasets", name=name.lower())
    data = dataset[0]
    name = name.lower()
    split_edge = randomsplit(dataset, name)
    #data = dataset[0]
    data.edge_index = to_undirected(split_edge["train"]["edge"].t())
    edge_index = data.edge_index
    data.num_nodes = data.x.shape[0]
 
    data.edge_weight = None 
    print(data.num_nodes, edge_index.max())
    # if data.edge_weight is None else data.edge_weight.view(-1).to(torch.float)
    # data = T.ToSparseTensor()(data)
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    
    print(name)

    if name=='fb':
        feature_embeddings=torch.load('datasets/fb_page/gnn_feature.pt')
        print(feature_embeddings)

        data.x = feature_embeddings

    data.max_x = -1

    print("dataset split ")
    for key1 in split_edge:
        for key2  in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])

    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t
    return data, split_edge



def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    #print(pos_test_pred)

    result = {}
    k_list = [1, 3, 10, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    # result_hit = {}
    for K in [1, 3, 10, 100]:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])


    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred.repeat(pos_train_pred.size(0), 1))
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1) )
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1) )
    
    # result_mrr = {}
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    # for K in [1,3,10, 100]:
    #     result[f'mrr_hit{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

   
    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])

    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])

    # print(train_pred)
    # print(train_true)
    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])

    
    return result
def train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None,addon= False,conn=False,avg_degree_embeddings=False,avg_degree_embeddings2={},hidden_channel:int=0,df=None,data_dict={},data_name=None):
    def penalty(posout, negout):
        scale = torch.ones_like(posout[[0]]).requires_grad_()
        loss = -F.logsigmoid(posout*scale).mean()-F.logsigmoid(-negout*scale).mean()
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(torch.square(grad))
    
    additional_features=None
    
    if alpha is not None:
        predictor.setalpha(alpha)

    
    
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    
    if data_name=='fb':negedge=negative_sampling(data.edge_index.to(pos_train_edge.device),num_nodes=22470)
    else:negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    
    #data.num_nodes=22470
    
    for perm in PermIterator(
            adjmask.device, adjmask.shape[0], batch_size
    ):
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t

        #print(data.x.shape)
        #print(adj.shape)
        h = model(data.x, adj)
        edge = pos_train_edge[:, perm]

        #data_fetch(conn, edges,data,avg_degree_embeddings,args,avg_degree_embeddings2=None)

        if addon:
            #additional_features=data_fetch(conn,edge,data,avg_degree_embeddings,hidden_channel,avg_degree_embeddings2)
            additional_features=data_fetch_prob(data.y, data_dict, edge)
            additional_features=additional_features.to(edge.device)

        pos_outs = predictor.multidomainforward(h,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs,additional=additional_features)

        pos_losss = -F.logsigmoid(pos_outs).mean()
        edge = negedge[:, perm]

        if addon:
            #data_fetch_prob(y, data_dict, edges)
            #additional_features=data_fetch(conn,edge,data,avg_degree_embeddings,hidden_channel,avg_degree_embeddings2)
            additional_features=data_fetch_prob(data.y,data_dict,edge)
            additional_features=additional_features.to(edge.device)

        neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs,additional=additional_features)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_losss + pos_losss
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss


@torch.no_grad()
def test(model, predictor, data, split_edge,  evaluator_hit, evaluator_mrr, batch_size,
         use_valedges_as_input,addon,df=None,data_dict={},flag=0,name=None):
    model.eval()
    predictor.eval()

    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = model(data.x, adj)

    pos_valid_pred = torch.cat([
        predictor(h, adj, pos_valid_edge[perm].t(),
        additional_features= (data_fetch_prob(data.y, data_dict, pos_valid_edge[perm].t()).to(pos_valid_edge.device)) if addon else None
        ).squeeze().cpu()
        for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False)
    ], dim=0)

    neg_valid_pred = torch.cat([
        predictor(h, adj, neg_valid_edge[perm].t(),
        additional_features= (data_fetch_prob(data.y, data_dict, neg_valid_edge[perm].t()).to(neg_valid_edge.device)) if addon else None
        ).squeeze().cpu()
        for perm in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], batch_size, False)
    ], dim=0)

    if use_valedges_as_input:
        adj = data.full_adj_t
        h = model(data.x, adj)

    pos_test_pred = torch.cat([
        predictor(h, adj, pos_test_edge[perm].t(),
        additional_features= (data_fetch_prob(data.y, data_dict, pos_test_edge[perm].t()).to(pos_test_edge.device)) if addon else None
        ).squeeze().cpu()
        for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False)
    ], dim=0)

    neg_test_pred = torch.cat([
        predictor(h, adj, neg_test_edge[perm].t(),
       additional_features= (data_fetch_prob(data.y, data_dict, neg_test_edge[perm].t()).to(neg_test_edge.device)) if addon else None
        ).squeeze().cpu()
        for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                 batch_size, False)
    ], dim=0)
    
    print('train valid_pos valid_neg test_pos test_neg', pos_valid_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_valid_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), h.cpu()]

    # --- SAVING LOGIC STARTS HERE ---
    if flag == 1:
        # Define ground truth labels
        pos_valid_truth = torch.ones(pos_valid_pred.shape[0], dtype=int)
        neg_valid_truth = torch.zeros(neg_valid_pred.shape[0], dtype=int)
        pos_test_truth = torch.ones(pos_test_pred.shape[0], dtype=int)
        neg_test_truth = torch.zeros(neg_test_pred.shape[0], dtype=int)

        # 1. Combine all edges
        all_edges = torch.cat([pos_valid_edge.cpu(), neg_valid_edge.cpu(), pos_test_edge.cpu(), neg_test_edge.cpu()], dim=0)

        # 2. Combine all raw model scores (logits)
        all_scores = torch.cat([pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred], dim=0).numpy()

        # 3. Combine all ground truth labels
        all_labels = torch.cat([pos_valid_truth, neg_valid_truth, pos_test_truth, neg_test_truth], dim=0).numpy()
        
        # 4. Create a list to identify the data split
        all_splits = (["valid"] * (len(pos_valid_edge) + len(neg_valid_edge))) + \
                     (["test"] * (len(pos_test_edge) + len(neg_test_edge)))

        # 5. Create the Essential Prediction DataFrame
        df_essential = pd.DataFrame({
            'node_u': all_edges[:, 0].numpy(),
            'node_v': all_edges[:, 1].numpy(),
            'label': all_labels,
            'model_score': all_scores,
            'split': all_splits
        })

        # 6. Add basic Classification Type (Optional, based on 0.0 logit threshold)
        # TP = Label 1 & Score > 0
        # FP = Label 0 & Score > 0
        # FN = Label 1 & Score <= 0
        # TN = Label 0 & Score <= 0
        df_essential['type'] = [
            ("TP" if s > 0 else "FN") if l == 1 else ("FP" if s > 0 else "TN")
            for l, s in zip(df_essential['label'], df_essential['model_score'])
        ]

        # 7. Save to CSV using the passed 'name' path
        if name:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(name)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            print(f"Saving prediction file to: {name}")
            df_essential.to_csv(name, index=False)
        else:
            print("Warning: No filename provided for saving results.")

    return result, score_emb


def parseargs():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--mplayers', type=int, default=1)
    parser.add_argument('--nnlayers', type=int, default=3)
    parser.add_argument('--ln', action="store_true")
    parser.add_argument('--lnnn', action="store_true")
    parser.add_argument('--res', action="store_true")
    parser.add_argument('--jk', action="store_true")
    parser.add_argument('--maskinput', action="store_true")
    parser.add_argument('--hiddim', type=int, default=32)
    parser.add_argument('--gnndp', type=float, default=0.3)
    parser.add_argument('--xdp', type=float, default=0.3)
    parser.add_argument('--tdp', type=float, default=0.3)
    parser.add_argument('--gnnedp', type=float, default=0.3)
    parser.add_argument('--predp', type=float, default=0.3)
    parser.add_argument('--preedp', type=float, default=0.3)
    parser.add_argument('--splitsize', type=int, default=-1)
    parser.add_argument('--gnnlr', type=float, default=0.0003)
    parser.add_argument('--prelr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--testbs', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--trndeg', type=int, default=-1)
    parser.add_argument('--tstdeg', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default="pubmed")
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    parser.add_argument('--model', choices=convdict.keys())
    parser.add_argument('--cndeg', type=int, default=-1)
    parser.add_argument('--save_gemb', action="store_true")
    parser.add_argument('--load', type=str)
    parser.add_argument('--cnprob', type=float, default=0)
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument("--learnpt", action="store_true")
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--increasealpha", action="store_true")
    parser.add_argument("--savex", action="store_true")
    parser.add_argument("--loadx", action="store_true")
    parser.add_argument("--loadmod", action="store_true")
    parser.add_argument("--savemod", action="store_true")


    ###
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--eval_steps', type=int, default=5)
    
    
    parser.add_argument('--addon', action='store_true',default=False)
    parser.add_argument('--cluster', action='store_true',default=False)
    parser.add_argument("--clustering", type=str, default="kmeans",
                        choices=["kmeans", "louvain", "spectral"],
                        help="Clustering method to use.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of clusters (only for kmeans/spectral).")


    # ADD THE NEW ARGUMENTS FOR LABEL NOISE HERE
    parser.add_argument('--noise_label', action='store_true', default=False,
                        help='Flag to add uniform noise to node labels.')
    parser.add_argument('--noise_rate', type=float, default=0.0,
                        help='Percentage of labels to flip (e.g., 0.1 for 10%). Used with --noise_label.')

    parser.add_argument('--uniform_addon', action='store_true', default=False,
                        help='Use uniform (uninformative) probabilities for the addon feature test.')

    
    args = parser.parse_args()
    return args


def main():
    args = parseargs()



    print(args, flush=True)
  
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs)
    }


    # data, split_edge,connection_probs,avg_degree_embeddings,degrees = loaddataset(args.dataset, args.use_valedges_as_input, args.load,args.components)

    # data = data.to(device)

    predfn = predictor_dict[args.predictor]

    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    ret = []
    
    for run in range(0, args.runs):

        run_start_time = time.time()

        if args.dataset=='fb':
            data, split_edge= loaddataset(args.dataset, args.use_valedges_as_input, args.load)
            if args.addon:
                df=pd.read_csv('datasets/fb_page/musae_facebook_target.csv')
                data_dict=get_fb_prob(df,split_edge)
            else:
                df=None
                data_dict={}
        else:
            data, split_edge= loaddataset(args.dataset, args.use_valedges_as_input, args.load)

            
            # find_optimal_k(split_edge, data,args.dataset)
            # exit()


            #data.edgeIndex=randomly_remove_edges(data.edge_index,args.components)
            if args.addon:
                if args.cluster:
                    if args.clustering == "kmeans":
                        #kmeans = KMeans(n_clusters=args.k, random_state=42).fit(data.x.cpu().numpy())
                        cluster_labels,_ = get_cluster_labels(split_edge, data, k=args.k, max_iters=100)
                    elif args.clustering == "louvain":
                        cluster_labels,_ = apply_louvain(data)
                    elif args.clustering == "spectral":
                        cluster_labels,_ = apply_spectral(data, args.k)
                    # print("Clustering with K-means with k=",args.k_means)
                    # cluster_labels,_ = get_cluster_labels(split_edge, data, k=args.k_means, max_iters=100)
                    #print(cluster_labels)
                    #print(data.y)
                # Save the cluster labels in data.y if you want
                    data.y = cluster_labels
                    print(data.y)

                    # cluster_distribution = np.bincount(cluster_labels.numpy())
                    # for cluster_id, count in enumerate(cluster_distribution):
                    #     print(f"Cluster {cluster_id}: {count} nodes")
                
                #print(data.y)
                if args.noise_label and args.noise_rate > 0:
                    print(f"Applying {args.noise_rate * 100:.2f}% uniform noise to labels...")
                    original_y = data.y.clone()
                    data.y = add_label_noise_manual(data.y, args.noise_rate)
                    
                    # Optional: Print how many labels were actually changed for verification
                    num_changed = (original_y != data.y).sum().item()
                    print(f"Flipped {num_changed} out of {len(data.y)} labels.")
                
                # UPDATE THIS LINE to pass the 'args' object
                data_dict = get_class_intersection_prob(data.y, split_edge, args)
                df=None
            else:
                df=None
                data_dict={}
        
        
        
        # # Check original shapes of features and labels
        # print("Original shape of data.x (node features):", data.x.shape)
        # #print("Shape of data.y (labels):", data.y.shape)

        # # Convert labels to a one-hot encoding for concatenation
        # #one_hot_labels = torch.nn.functional.one_hot(data.y, num_classes=data.y.max().item() + 1).float()

        # # Concatenate node features with one-hot encoded labels
        # #data.x = torch.cat([data.x, one_hot_labels], dim=1)
        # #data.x = torch.cat([data.x, data.y], dim=1)

        # # Check new shape of data.x
        # print("New shape of data.x (features + labels):", data.x.shape)
        
        data = data.to(device)

        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        init_seed(seed)

        save_path = args.output_dir+'/lr'+str(args.gnnlr) + '_drop' + str(args.gnndp) + '_l2'+ str(args.l2) + '_numlayer' + str(args.mplayers)+ '_numPredlay' + str(args.nnlayers) +'_dim'+str(args.hiddim) + '_'+ 'best_run_'+str(seed)


        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
       
        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
       
        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': predictor.parameters(), 'lr': args.prelr}],  weight_decay=args.l2)
        
        best_valid = 0
        kill_cnt = 0
        avg_degree_embeddings2={}
        for epoch in range(1, 1 + args.epochs):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput, [], alpha,addon=args.addon,df=df,data_dict=data_dict,data_name=args.dataset)#,conn=connection_probs,avg_degree_embeddings=avg_degree_embeddings,hidden_channel=args.hiddim)
            
            #avg_degree_embeddings2= degree_connection_probabilities_and_avg_embeddings2(data.edge_index,degrees,node_embd)
            # print(f"trn time {time.time()-t1:.2f} s", flush=True)
            
            t1 = time.time()
            if epoch % args.eval_steps == 0:
                results, score_emb = test(model, predictor, data, split_edge,  evaluator_hit, evaluator_mrr,
                                args.testbs, args.use_valedges_as_input,addon=args.addon,df=df,data_dict=data_dict)#conn=connection_probs,avg_degree_embeddings=avg_degree_embeddings,hidden_channel=args.hiddim,avg_degree_embeddings2=avg_degree_embeddings2)
                # print(f"test time {time.time()-t1:.2f} s")
            
                
                for key, result in results.items():
                    _, valid_hits, test_hits = result

                   
                    loggers[key].add_result(run, result)
                        
                    print(key)
                    log_print.info(
                        f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                print('---', flush=True)

                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max().item()

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if args.save:

                        save_emb(score_emb, save_path)
        
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
            #break



        if args.addon:
            gbb="addone"
        else:gbb='none'
        # 1. Get the name of the current script (e.g., "main_ncn")
        # sys.argv[0] gets the script path, basename gets the filename, splitext removes .py
        script_name = "ncn"
        
        # 2. Construct the dynamic path
        # Structure: all_edges / <dataset> / <script_name> / run_<run_id>.csv
        # Example: all_edges/cora/main_ncn/run_0.csv
        output_filename = f"edges_analysis/{args.dataset.lower()}/{script_name}/run_{run}.csv"

        # 3. Call test with flag=1 and the new filename
        results, score_emb = test(
            model, predictor, data, split_edge, evaluator_hit, evaluator_mrr,
            args.testbs, args.use_valedges_as_input, 
            addon=args.addon, 
            df=df, 
            data_dict=data_dict, 
            flag=1, 
            name=output_filename  # Pass the dynamic name here
        )

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

        run_end_time = time.time()
        print(f"Run {run + 1} completed in {run_end_time - run_start_time:.5f} seconds.")
        print("-" * 60) # Separator for readability
    
    result_all_run = {}
    for key in loggers.keys():
        print(key)
        
        best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()

        if key == eval_metric:
            best_metric_valid_str = best_metric
            best_valid_mean_metric = best_valid_mean
            
        if key == 'AUC':
            best_auc_valid_str = best_metric
            best_auc_metric = best_valid_mean

        result_all_run[key] = [mean_list, var_list]
        
    
    print(best_metric_valid_str +' ' +best_auc_valid_str)

    return best_valid_mean_metric, best_auc_metric, result_all_run

 

if __name__ == "__main__":
    main()
  
