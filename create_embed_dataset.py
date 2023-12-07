import numpy as np
import models
from dataset import UPFD
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch
import h5py
import argparse

dict_models = {'original':models.Original_model}

def create_embedding_dataset(model_checkpoint,model_class):
    model = dict_models[model_class]
    checkpoint = torch.load(model_checkpoint)

    description = checkpoint['description_experiment']
    print(description)

    train_dataset = UPFD('.',
                         description['dataset_name'],
                         description['features'],
                         'train')
    test_dataset = UPFD('.',
                       description['dataset_name'],
                       description['features'],
                       'test')
    model = model(dataset=train_dataset)

    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print('Model retrieved at epoch {}'.format(epoch))
    test_loss = checkpoint['loss']
    test_accuracy = checkpoint['accuracy']
    print('loss {:.3f}, accuracy {:.3f}'.format(test_loss,test_accuracy))
    
    device = 'cpu'
    
    
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=description['batch_size'])
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=description['batch_size'])
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
        list_out_train = np.concatenate(list_out_train)
        
        
    
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
        list_out_test = np.concatenate(list_out_test)

        with h5py.File("embeddings_graphs.hdf5", "w") as f:
            f.create_dataset("test_embeds",data = list_out_test)
            f.create_dataset("train_embeds",data = list_out_train)

                
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Params for training models")

    # Common params
    parser.add_argument("--model_checkpoint", help="Directory to save the models",
                        type =str,
                        default='models/best_model__politifact_profile_8.pt')
    
    args  = parser.parse_args()
    create_embedding_dataset(model_checkpoint=args.model_checkpoint,
                             model_class='original')
