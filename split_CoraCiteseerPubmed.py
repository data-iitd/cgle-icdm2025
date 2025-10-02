### exampled code to split small datasets

import torch
from torch_geometric.datasets import Planetoid,HeterophilousGraphDataset,Amazon,Coauthor,AttributedGraphDataset,CitationFull
from torch_geometric.utils import coalesce, to_undirected
from torch_geometric.data import InMemoryDataset, Data
import os

class MyCustomDataset(InMemoryDataset):
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
        node_features = torch.load(self.raw_paths[1])  # 'node_features.pt'
        edge_list = torch.load(self.raw_paths[0])      # 'edge_list.pt'

        # Create a Data object
        data = Data(x=node_features, edge_index=edge_list.contiguous())

        # Optionally apply pre-transformations
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_list = [data]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



def save(file_name, data_name, edge):
    # Ensure the directory exists
    directory = f'data_splits/{data_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Write the edges to the file
    with open(f'{directory}/{file_name}.txt', 'w') as f:
        for i in range(edge.size(1)):
            s, t = edge[0][i].item(), edge[1][i].item()
            f.write(f'{s}\t{t}\n')
            f.flush()


data_name = 'Cora'  #Include your dataset here

dataset = CitationFull(root=".",name=data_name) #Include

data = dataset[0]

### get unique edges
edge_index = data.edge_index
edge_index = to_undirected(edge_index)
edge_index = coalesce(edge_index)
mask = edge_index[0] <= edge_index[1]
edge_index = edge_index[:, mask]

### split 
perm = torch.randperm(edge_index.size(1))
test_pos_len = int(len(perm)*0.1)
valid_pos_len = int(len(perm)*0.05)
valid_pos = edge_index[:,perm[:valid_pos_len]]
test_pos = edge_index[:,perm[valid_pos_len:valid_pos_len+test_pos_len]]
train_pos = edge_index[:, perm[test_pos_len+valid_pos_len:]]

### to generate negatives 
nodenum = data.x.size(0)
#nodenum=22470

edge_dict = {}
for i in range(edge_index.size(1)):
    s, t = edge_index[0][i].item(), edge_index[1][i].item()
    if s not in edge_dict: edge_dict[s] = set()
    if t not in edge_dict: edge_dict[t] = set()
    edge_dict[s].add(t)
    edge_dict[t].add(s)

### negatives should not be the positive edges
valid_neg = []
for i in range(valid_pos.size(1)):
    src = torch.randint(0, nodenum, (1,)).item()
    dst = torch.randint(0, nodenum, (1,)).item()
    while dst in edge_dict[src] or src in edge_dict[dst]:
        src = torch.randint(0, nodenum, (1,)).item()
        dst = torch.randint(0, nodenum, (1,)).item()

    valid_neg.append([src, dst])


test_neg = []
for i in range(test_pos.size(1)):
    src = torch.randint(0, nodenum, (1,)).item()
    dst = torch.randint(0, nodenum, (1,)).item()
    while dst in edge_dict[src] or src in edge_dict[dst]:
        src = torch.randint(0, nodenum, (1,)).item()
        dst = torch.randint(0, nodenum, (1,)).item()

    test_neg.append([src, dst])

valid_neg = torch.tensor(valid_neg).t()
test_neg = torch.tensor(test_neg).t()

### save data
save('train_pos', data_name, train_pos)
save('valid_pos', data_name, valid_pos)
save('valid_neg', data_name, valid_neg)

save('test_pos', data_name, test_pos)
save('test_neg', data_name, test_neg)

print(data.x.shape)

# feature_file_path = 'dataset/' + data_name + '/gnn_feature.pt'
# torch.save(data.x, feature_file_path)