import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import torch
import networkx as nx
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, to_networkx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Planetoid, HeterophilousGraphDataset, AttributedGraphDataset, Coauthor, CitationFull, Actor, WikipediaNetwork
from ogb.linkproppred import PygLinkPropPredDataset

# Assuming LINKXDataset is available in your environment or torch_geometric.datasets
try:
    from torch_geometric.datasets import LINKXDataset
except ImportError:
    pass # Handle if not available

from tqdm import tqdm

# ==========================================
# 1. USER PROVIDED LOADING LOGIC
# ==========================================
cwd = os.getcwd()

class MyCustomDataset(InMemoryDataset): 
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyCustomDataset, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self): return ['node_features.pt', 'edge_list.pt']
    @property
    def processed_file_names(self): return ['data.pt']
    def download(self): pass
    def process(self):
        node_features = torch.load(self.raw_paths[0])
        edge_list = torch.load(self.raw_paths[1])
        data = Data(x=node_features, edge_index=edge_list.contiguous())
        if self.pre_transform is not None: data = self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

def randomsplit(dataset, data_name, k=0):
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    train_pos, valid_pos, test_pos = [], [], []
    valid_neg, test_neg = [], []
    node_set = set()
    dir_path = "/home/ankit/research/CGLE_code/CGLE2/CGLE/CGLE"
    
    # Check if files exist
    check_path = dir_path+'/data_splits' + '/{}/{}_pos.txt'.format(data_name, 'train')
    if not os.path.exists(check_path):
        raise FileNotFoundError(f"Split file not found at: {check_path}. Make sure you are running this from the correct root directory.")

    for split in ['train', 'test', 'valid']:
        path = dir_path+'/data_splits' + '/{}/{}_pos.txt'.format(data_name, split)
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            node_set.add(sub); node_set.add(obj)
            if sub == obj: continue
            if split == 'train': train_pos.append((sub, obj))
            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    
    for split in ['test', 'valid']:
        path = dir_path+'/data_splits' + '/{}/{}_neg.txt'.format(data_name, split)
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            if split == 'valid': valid_neg.append((sub, obj))
            if split == 'test': test_neg.append((sub, obj))

    split_edge['train']['edge'] = torch.tensor(train_pos)
    split_edge['valid']['edge'] = torch.tensor(valid_pos)
    split_edge['valid']['edge_neg'] = torch.tensor(valid_neg)
    split_edge['test']['edge'] = torch.tensor(test_pos)
    split_edge['test']['edge_neg'] = torch.tensor(test_neg)
    return split_edge

def loaddataset(name, use_valedges_as_input, load=None, k=0):
    if name=='fb': dataset = MyCustomDataset(root="datasets/fb_page")
    elif name in ["ppi",'TWeibo','Flickr','Facebook']: dataset=AttributedGraphDataset(root="datasets", name=name)
    elif name in['CS','Physics']: dataset=Coauthor(root="datasets",name=name)
    elif name in ["Roman-empire", "Amazon-ratings","Questions","Minesweeper", "Tolokers"]: dataset = HeterophilousGraphDataset(root="datasets", name=name.lower())
    elif name=='DBLP': dataset=CitationFull(root='datasets',name=name)
    elif name in ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"]: dataset=LINKXDataset(root='datasets', name=name)
    elif name=="actor": dataset=Actor(root="datasets")
    elif name in ["chameleon", "crocodile", "squirrel"]: dataset=WikipediaNetwork(root="datasets",name=name)
    else: dataset = Planetoid(root="datasets", name=name.lower())
    
    data = dataset[0]
    name = name.lower()
    split_edge = randomsplit(dataset, name)
    
    # Construct Training Graph
    data.edge_index = to_undirected(split_edge["train"]["edge"].t())
    
    data.num_nodes = data.x.shape[0]
    data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).to_symmetric().coalesce()
    
    if name=='fb':
        feature_embeddings=torch.load('datasets/fb_page/gnn_feature.pt')
        data.x = feature_embeddings

    return data, split_edge

# ==========================================
# 2. FEATURE CALCULATION HELPERS
# ==========================================

def calculate_structural_features(G, pairs):
    """
    G: NetworkX graph (Training Graph)
    pairs: List of tuples (u, v)
    """
    degrees = []
    cns = []
    sps = []
    
    # Pre-compute degrees dictionary for speed
    deg_dict = dict(G.degree())
    
    print(f"  > Calculating features for {len(pairs)} edges...")
    
    for u, v in tqdm(pairs, leave=False):
        u, v = int(u), int(v)
        
        # 1. Degree Sum (Sparsity)
        d_u = deg_dict.get(u, 0)
        d_v = deg_dict.get(v, 0)
        degrees.append(d_u + d_v)
        
        # 2. Common Neighbors (Heuristic)
        # Note: cn is only defined if nodes exist in G
        if G.has_node(u) and G.has_node(v):
            cn_count = len(list(nx.common_neighbors(G, u, v)))
        else:
            cn_count = 0
        cns.append(cn_count)
        
        # 3. Shortest Path (Long Range)
        # We use a cutoff of 4 to speed up. 5 means >= 5 or disconnected
        try:
            if G.has_node(u) and G.has_node(v):
                sp = nx.shortest_path_length(G, source=u, target=v)
            else:
                sp = 99
        except nx.NetworkXNoPath:
            sp = 99
        
        sps.append(sp)
        
    return degrees, cns, sps

# ==========================================
# 3. MAIN ANALYSIS LOOP
# ==========================================

# CONFIGURATION
DATASETS = ["facebook", "physics", "roman_empire", "amazon_ratings",
                   "questions", "chameleon"] # Add your datasets here
MODELS = ["ncn", "ncnc", "cgle_ncn", "cgle_ncnc"] # Your folder names
BASE_DIR = ""
NUM_RUNS = 10 

for dataset_name in DATASETS:
    print(f"\n================ processing dataset: {dataset_name} ================")
    
    # --- A. LOAD REAL GRAPH STRUCTURE ---
    try:
        print("  > Loading Dataset and constructing Training Graph...")
        # Use user's function to get the training graph
        data, _ = loaddataset(dataset_name, use_valedges_as_input=False)
        
        # Convert to NetworkX for feature calculation
        # Data.edge_index contains ONLY training edges here, which is correct.
        G = to_networkx(data, to_undirected=True)
        print(f"  > Graph Loaded. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        
    except Exception as e:
        print(f"  ! Error loading dataset {dataset_name}: {e}")
        continue

    dataset_dfs = [] 
    
    # --- B. PROCESS MODELS ---
    for model_name in MODELS:
        print(f"  > Processing Model: {model_name}")
        
        for run_id in range(NUM_RUNS):
            file_path = os.path.join(BASE_DIR, dataset_name, model_name, f"run_{run_id}.csv")
            
            if not os.path.exists(file_path):
                print(f"    ! Results file not found: {file_path}. Skipping.")
                continue
            
            # Load Results
            df = pd.read_csv(file_path)
            if 'split' in df.columns:
                df = df[df['split'] == 'test'].copy()
            
            # 3. Add Metadata
            df['model'] = model_name
            df['run_id'] = run_id
            
            # --- C. CALCULATE REAL FEATURES ---
            # Extract pairs from dataframe
            pairs = list(zip(df['node_u'].values, df['node_v'].values))
            
            # Calculate features using the loaded NetworkX graph G
            # Note: This might be slow for large datasets like Pubmed.
            # Optimization: If the test set is IDENTICAL across runs, calculate once outside the loop.
            # Assuming randomness in test/neg sampling might vary slightly per run, we calculate inside.
            degs, cns, sps = calculate_structural_features(G, pairs)
            
            df['degree_sum'] = degs
            df['common_neighbors'] = cns
            df['shortest_path'] = sps

            # --- D. BINNING ---
            # Bin Common Neighbors
            bins_cn = [-1, 0, 2, 5, 1000]
            labels_cn = ['0', '1-2', '3-5', '6+']
            df['cn_bin'] = pd.cut(df['common_neighbors'], bins=bins_cn, labels=labels_cn)

            # Bin Degree
            bins_deg = [0, 5, 20, 1000] 
            labels_deg = ['Sparse (<5)', 'Normal (5-20)', 'Hubs (>20)']
            df['deg_bin'] = pd.cut(df['degree_sum'], bins=bins_deg, labels=labels_deg)
            
            # Bin Shortest Path
            # Group anything > 4 into "Long"
            df['sp_bin'] = df['shortest_path'].apply(lambda x: str(x) if x <= 4 else ">4")

            dataset_dfs.append(df)

    # --- E. PLOTTING ---
    if not dataset_dfs:
        print("  ! No data found to plot.")
        continue

    full_df = pd.concat(dataset_dfs, ignore_index=True)
    pos_links = full_df[full_df['label'] == 1].copy()
    
    # Define correctness
    if 'type' in pos_links.columns:
        pos_links['is_correct'] = pos_links['type'] == 'TP'
    else:
        pos_links['is_correct'] = pos_links['model_score'] > 0 

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("bright", n_colors=len(MODELS)) 
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Structural Analysis: {dataset_name.upper()}', fontsize=16)

    # Plot 1: Common Neighbors
    sns.barplot(data=pos_links, x='cn_bin', y='is_correct', hue='model', ax=axes[0], palette=palette, capsize=.1, errorbar='sd')
    axes[0].set_title("Recall by Common Neighbors")
    
    # Plot 2: Degree
    sns.barplot(data=pos_links, x='deg_bin', y='is_correct', hue='model', ax=axes[1], palette=palette, capsize=.1, errorbar='sd')
    axes[1].set_title("Recall by Node Degree")

    # Plot 3: Shortest Path
    sns.barplot(data=pos_links, x='sp_bin', y='is_correct', hue='model', ax=axes[2], palette=palette, capsize=.1, errorbar='sd')
    axes[2].set_title("Recall by Shortest Path")
    axes[2].set_xlabel("Hop Distance (Training Graph)")

    output_plot_dir = "analysis_plots_real"
    os.makedirs(output_plot_dir, exist_ok=True)
    save_name = f"{output_plot_dir}/{dataset_name}_analysis.png"
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    
    print(f"  > Plot saved to {save_name}")

print("\nAnalysis Complete.")