
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool
from torch.nn import Linear,Softmax
from dataset import UPFD
### TO DO : define the dataset properly

class Original_model(torch.nn.Module):
    def __init__(self,dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.linear1 = Linear(64,32)
        self.linear2 = Linear(32,2)
        self.softmax = Softmax(dim=0)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear1(x)
        x = self.linear2(x)
        return self.softmax(x)
#dataset_upfd = UPFD('.','politifact','profile','train')
