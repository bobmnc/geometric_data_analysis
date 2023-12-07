import numpy as np
import models
from dataset import UPFD
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch
import h5py

dict_models = {'original':models.Original_model()}

def create_embedding_dataset(model_checkpoint,model_class):
    model = dict_models[model_class]
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print('Model retrieved at epoch {}'.format(epoch))
    test_loss = model_checkpoint['loss']
    test_accuracy = model_checkpoint['accuracy']
    print('loss {:.3f}, accuracy {:.3f}'.format(test_loss,test_accuracy))
    description = model_checkpoint['description_experiment']
    device = 'cpu'
    train_dataset = UPFD('.',
                         description.dataset_name,
                         description.features,
                         'train')
    test_dataset = UPFD('.',
                        description.dataset,
                        description.features,
                        'test')
    
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=description.batch_size)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=description.batch_size)
    model.eval()
    with torch.no_grad():
        list_out_train = list()
        for batch in tqdm(train_dataloader,desc='Embedding train set'):
                y = batch.y
                batch = batch.to(device)
                y = y.to(device)
                out = model(batch,
                            return_embed=True)
                out = out.numpy()
                list_out_train.append(out)
        list_out_train = np.array(list_out_train)
        list_out_train = np.reshape(list_out_train,(1,-1))
        with h5py.File("embeddings_graphs.hdf5", "w") as f:
            f.create_dataset("train_embeds",data = list_out_train)
    
    with torch.no_grad():
        list_out_test = list()
        for batch in tqdm(test_dataloader,desc='Embedding test set'):
                y = batch.y
                batch = batch.to(device)
                y = y.to(device)
                out = model(batch,
                            return_embed=True)
                out = out.numpy()
                list_out_test.append(out)
        list_out_test = np.array(list_out_test)
        list_out_test = np.reshape(list_out_test,(1,-1))
        with h5py.File("embeddings_graphs.hdf5", "w") as f:
            f.create_dataset("test_embeds",data = list_out_test)

                

