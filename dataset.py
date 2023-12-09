import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce, cumsum


class UPFD(InMemoryDataset):
    r"""The tree-structured fake news propagation graph classification dataset
    from the `"User Preference-aware Fake News Detection"
    <https://arxiv.org/abs/2104.12259>`_ paper.
    It includes two sets of tree-structured fake & real news propagation graphs
    extracted from Twitter.
    For a single graph, the root node represents the source news, and leaf
    nodes represent Twitter users who retweeted the same root news.
    A user node has an edge to the news node if and only if the user retweeted
    the root news directly.
    Two user nodes have an edge if and only if one user retweeted the root news
    from the other user.
    Four different node features are encoded using different encoders.
    Please refer to `GNN-FakeNews
    <https://github.com/safe-graph/GNN-FakeNews>`_ repo for more details.

    .. note::

        For an example of using UPFD, see `examples/upfd.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        upfd.py>`_.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the graph set (:obj:`"politifact"`,
            :obj:`"gossipcop"`).
        feature (str): The node feature type (:obj:`"profile"`, :obj:`"spacy"`,
            :obj:`"bert"`, :obj:`"content"`).
            If set to :obj:`"profile"`, the 10-dimensional node feature
            is composed of ten Twitter user profile attributes.
            If set to :obj:`"spacy"`, the 300-dimensional node feature is
            composed of Twitter user historical tweets encoded by
            the `spaCy word2vec encoder
            <https://spacy.io/models/en#en_core_web_lg>`_.
            If set to :obj:`"bert"`, the 768-dimensional node feature is
            composed of Twitter user historical tweets encoded by the
            `bert-as-service <https://github.com/hanxiao/bert-as-service>`_.
            If set to :obj:`"content"`, the 310-dimensional node feature is
            composed of a 300-dimensional "spacy" vector plus a
            10-dimensional "profile" vector.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    url = 'https://docs.google.com/uc?export=download&id={}&confirm=t'

    ids = {
        'politifact': '1KOmSrlGcC50PjkvRVbyb_WoWHVql06J-',
        'gossipcop': '1VskhAQ92PrT4sWEKQ2v2-AJhEcpp4A81',
    }

    def __init__(
        self,
        root: str,
        name: str,
        feature: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.root = root
        self.name = name
        if isinstance(feature,list) and len(feature)==1:
            feature = feature[0]
        self.feature = feature
        
        if isinstance(self.feature,list):
            path_dir = osp.join(self.root, self.name, 'processed')
            features_str = str()
            for ft in self.feature:
                features_str+=str(ft)
            path_dir = osp.join(self.root, self.name, 'processed', features_str)
        else : 
            path_dir =  osp.join(self.root, self.name, 'processed', self.feature)
        self.path_dir_processed = path_dir
        print(path_dir)
        super().__init__(root, transform, pre_transform, pre_filter)
                         #force_reload=force_reload)

        assert split in ['train', 'val', 'test']
        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.load(path)
        

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return self.path_dir_processed
    @property
    def raw_file_names(self) -> List[str]:
        if isinstance(self.feature,list):

            return [
                'node_graph_id.npy', 'graph_labels.npy', 'A.txt', 'train_idx.npy',
                'val_idx.npy', 'test_idx.npy']\
                    +[f'new_{feature}_feature.npz' for feature in self.feature]
        else :
            return [
                'node_graph_id.npy', 'graph_labels.npy', 'A.txt', 'train_idx.npy',
                'val_idx.npy', 'test_idx.npy', f'new_{self.feature}_feature.npz'
            ]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        path = download_url(self.url.format(self.ids[self.name]), self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        if isinstance(self.feature, list):
            list_x = []
            edge_indices_list = []  # To store edge indices for each feature
            for ft in self.feature:
                x = sp.load_npz(osp.join(self.raw_dir, f'new_{ft}_feature.npz'))
                x = torch.from_numpy(x.todense()).to(torch.float)
                list_x.append(x)

                edge_index = read_txt_array(osp.join(self.raw_dir, 'A.txt'), sep=',',
                                            dtype=torch.long).t()
                edge_index = coalesce(edge_index, num_nodes=x.size(0))
                edge_indices_list.append(edge_index)

            x = torch.cat(list_x, dim=1)
            print('final_x shape', x.shape)

            # Coalesce edge indices from all features
            edge_index = torch.cat(edge_indices_list, dim=1)
            edge_index = coalesce(edge_index, num_nodes=x.size(0))
        else:
            if self.feature == 'no_feature':
                ## if we want the model to look just at the shape of the graph
                dataset_size = {'politifact':41054,
                                'gossipcop':314_262 }
                x = torch.ones([dataset_size[self.name],
                                1],dtype=torch.float)
            else : 
                x = sp.load_npz(
                    osp.join(self.raw_dir, f'new_{self.feature}_feature.npz'))
                x = torch.from_numpy(x.todense()).to(torch.float)

            edge_index = read_txt_array(osp.join(self.raw_dir, 'A.txt'), sep=',',
                                        dtype=torch.long).t()
            edge_index = coalesce(edge_index, num_nodes=x.size(0))
        y = np.load(osp.join(self.raw_dir, 'graph_labels.npy'))
        y = torch.from_numpy(y).to(torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

        batch = np.load(osp.join(self.raw_dir, 'node_graph_id.npy'))
        batch = torch.from_numpy(batch).to(torch.long)

        node_slice = cumsum(batch.bincount())
        edge_slice = cumsum(batch[edge_index[0]].bincount())
        graph_slice = torch.arange(y.size(0) + 1)
        self.slices = {
            'x': node_slice,
            'edge_index': edge_slice,
            'y': graph_slice
        }
        edge_index -= node_slice[batch[edge_index[0]]].view(1, -1)
        self.data = Data(x=x, edge_index=edge_index, y=y)

        for path, split in zip(self.processed_paths, ['train', 'val', 'test']):
            idx = np.load(osp.join(self.raw_dir, f'{split}_idx.npy')).tolist()
            data_list = [self.get(i) for i in idx]
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            self.save(data_list, path)



    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, name={self.name}, '
                f'feature={self.feature})')




class UPFD_degradation(InMemoryDataset):
    r"""
    Version dégradé du dataset on choisit de garder seulement un pourcentage des
    noeuds par graph et on ne garde que les noeuds les plus récents 
    The tree-structured fake news propagation graph classification dataset
    from the `"User Preference-aware Fake News Detection"
    <https://arxiv.org/abs/2104.12259>`_ paper.
    It includes two sets of tree-structured fake & real news propagation graphs
    extracted from Twitter.
    For a single graph, the root node represents the source news, and leaf
    nodes represent Twitter users who retweeted the same root news.
    A user node has an edge to the news node if and only if the user retweeted
    the root news directly.
    Two user nodes have an edge if and only if one user retweeted the root news
    from the other user.
    Four different node features are encoded using different encoders.
    Please refer to `GNN-FakeNews
    <https://github.com/safe-graph/GNN-FakeNews>`_ repo for more details.

    .. note::

        For an example of using UPFD, see `examples/upfd.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        upfd.py>`_.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the graph set (:obj:`"politifact"`,
            :obj:`"gossipcop"`).
        feature (str): The node feature type (:obj:`"profile"`, :obj:`"spacy"`,
            :obj:`"bert"`, :obj:`"content"`).
            If set to :obj:`"profile"`, the 10-dimensional node feature
            is composed of ten Twitter user profile attributes.
            If set to :obj:`"spacy"`, the 300-dimensional node feature is
            composed of Twitter user historical tweets encoded by
            the `spaCy word2vec encoder
            <https://spacy.io/models/en#en_core_web_lg>`_.
            If set to :obj:`"bert"`, the 768-dimensional node feature is
            composed of Twitter user historical tweets encoded by the
            `bert-as-service <https://github.com/hanxiao/bert-as-service>`_.
            If set to :obj:`"content"`, the 310-dimensional node feature is
            composed of a 300-dimensional "spacy" vector plus a
            10-dimensional "profile" vector.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    url = 'https://docs.google.com/uc?export=download&id={}&confirm=t'

    ids = {
        'politifact': '1KOmSrlGcC50PjkvRVbyb_WoWHVql06J-',
        'gossipcop': '1VskhAQ92PrT4sWEKQ2v2-AJhEcpp4A81',
    }

    def __init__(
        self,
        root: str,
        name: str,
        feature: str,
        split: str = "train",
        drop_percentage: int = 20,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.root = root
        self.name = name
        if isinstance(feature,list) and len(feature)==1:
            feature = feature[0]
        self.feature = feature
        self.drop_percentage = drop_percentage
        
        if isinstance(self.feature,list):
            path_dir = osp.join(self.root, self.name, 'processed')
            features_str = str()
            for ft in self.feature:
                features_str+=str(ft)
            path_dir = osp.join(self.root, self.name, 'processed', features_str)
        else : 
            path_dir =  osp.join(self.root, self.name, 'processed', self.feature)
        self.path_dir_processed = path_dir
        print(path_dir)
        super().__init__(root, transform, pre_transform, pre_filter)
                         #force_reload=force_reload)

        assert split in ['train', 'val', 'test']
        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.load(path)
        

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return self.path_dir_processed
    @property
    def raw_file_names(self) -> List[str]:
        if isinstance(self.feature,list):

            return [
                'node_graph_id.npy', 'graph_labels.npy', 'A.txt', 'train_idx.npy',
                'val_idx.npy', 'test_idx.npy']\
                    +[f'new_{feature}_feature.npz' for feature in self.feature]
        else :
            return [
                'node_graph_id.npy', 'graph_labels.npy', 'A.txt', 'train_idx.npy',
                'val_idx.npy', 'test_idx.npy', f'new_{self.feature}_feature.npz'
            ]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        path = download_url(self.url.format(self.ids[self.name]), self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        if isinstance(self.feature, list):
            list_x = []
            edge_indices_list = []  # To store edge indices for each feature
            for ft in self.feature:
                x = sp.load_npz(osp.join(self.raw_dir, f'new_{ft}_feature.npz'))
                x = torch.from_numpy(x.todense()).to(torch.float)
                list_x.append(x)

                edge_index = read_txt_array(osp.join(self.raw_dir, 'A.txt'), sep=',',
                                            dtype=torch.long).t()
                edge_index = coalesce(edge_index, num_nodes=x.size(0))
                edge_indices_list.append(edge_index)

            x = torch.cat(list_x, dim=1)
            print('final_x shape', x.shape)

            # Coalesce edge indices from all features
            edge_index = torch.cat(edge_indices_list, dim=1)
            edge_index = coalesce(edge_index, num_nodes=x.size(0))
        else:
            if self.feature == 'no_feature':
                ## if we want the model to look just at the shape of the graph
                dataset_size = {'politifact':41054,
                                'gossipcop':314_262 }
                x = torch.ones([dataset_size[self.name],
                                1],dtype=torch.float)
            else : 
                x = sp.load_npz(
                    osp.join(self.raw_dir, f'new_{self.feature}_feature.npz'))
                x = torch.from_numpy(x.todense()).to(torch.float)

            edge_index = read_txt_array(osp.join(self.raw_dir, 'A.txt'), sep=',',
                                        dtype=torch.long).t()
            edge_index = coalesce(edge_index, num_nodes=x.size(0))
        

        graph_ids = np.load(osp.join(self.raw_dir, 'node_graph_id.npy'))
        graph_ids = torch.from_numpy(graph_ids)
        num_graphs = int(graph_ids.max()) + 1

        new_indices = {}
        current_index = 0
        index_to_keep = list()
        for graph_id in range(num_graphs):
            nodes = torch.nonzero(torch.tensor(graph_ids == graph_id)).view(-1)
            num_nodes = nodes.size(0)
            num_nodes_to_keep = int((100 - self.drop_percentage) * num_nodes / 100)

            index_to_keep+=[nodes[:num_nodes_to_keep]]

            for i, node in enumerate(nodes[:num_nodes_to_keep]):
                new_indices[node.item()] = current_index + i
            current_index += num_nodes_to_keep
            # Update edge_index for the current graph
        index_to_keep = torch.cat(index_to_keep)
        edge_index_path = osp.join(self.raw_dir, 'A.txt')
        edge_index = read_txt_array(edge_index_path, sep=',', dtype=torch.long).t()

        # Update edge_index according to the new node indices
        mask = torch.tensor([(edge_index[0,k] in index_to_keep) and (edge_index[1,k] in index_to_keep) for k in range(edge_index.size(1))])
        edge_index = edge_index[:,mask]#.reshape(2, -1)
        for old_idx, new_idx in new_indices.items():
            edge_index[edge_index == old_idx] = new_idx

        # Update x and other attributes using new_indices mapping
        x = x[index_to_keep]

        y = np.load(osp.join(self.raw_dir, 'graph_labels.npy'))
        y = torch.from_numpy(y).to(torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)
        

        batch = np.load(osp.join(self.raw_dir, 'node_graph_id.npy'))
        batch = torch.from_numpy(batch).to(torch.long)
        batch = batch[index_to_keep]

        node_slice = cumsum(batch.bincount())
        edge_slice = cumsum(batch[edge_index[0]].bincount())
        graph_slice = torch.arange(y.size(0) + 1)
        self.slices = {
            'x': node_slice,
            'edge_index': edge_slice,
            'y': graph_slice
        }
        edge_index -= node_slice[batch[edge_index[0]]].view(1, -1)
        self.data = Data(x=x, edge_index=edge_index, y=y)

        for path, split in zip(self.processed_paths, ['train', 'val', 'test']):
            idx = np.load(osp.join(self.raw_dir, f'{split}_idx.npy')).tolist()
            data_list = [self.get(i) for i in idx]
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            self.save(data_list, path)



    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, name={self.name}, '
                f'feature={self.feature})')