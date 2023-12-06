
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool
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
        self.pooling = global_mean_pool
        
    def forward(self, data,return_embed = False):
        x, edge_index = data.x, data.edge_index
        batch_vev = data.batch

        x = self.conv1(x,
                    edge_index)
        x = F.relu(x)
        x = F.dropout(x,
                       training=self.training)
        x = self.conv2(x, 
                       edge_index)
        x = self.pooling(x,
                         batch=batch_vev)
        if return_embed :
            return x
        x = self.linear1(x)
        x = self.linear2(x)
        return self.softmax(x)
#dataset_upfd = UPFD('.','politifact',['profile','content'],'train')
