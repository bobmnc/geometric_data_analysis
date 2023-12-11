import torch.nn as nn
from torch import optim
from models import Original_model,Deepset
from dataset import UPFD
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch
import argparse
import os.path as osp
import sklearn.metrics as metrics
from utils import get_file_name_desc


def train(model,train_dataloader,test_dataloader,N_epochs,save_path,description_exp,device='cuda:0'):
    loss_fn = nn.CrossEntropyLoss()
    best_test_loss = 10**9
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    for epoch in tqdm(range(N_epochs)):
        model.train()
        tot_loss = 0
        n_corrects = 0
        n_tot = 0
        for batch in train_dataloader:
            y = batch.y
            optimizer.zero_grad()
            batch = batch.to(device)
            y = y.to(device)
            out = model(batch)
            loss = loss_fn(out,y)
            loss.backward()
            optimizer.step()
            tot_loss +=loss.cpu().detach().item()
            pred = out.argmax(dim=-1)
            n_tot+= len(pred)
            n_corrects+= (pred==y).sum()
        if epoch%5==0:
            print('Epoch {} loss : {:.3f} accuracy :{} '.format(epoch,
                                                                loss/n_tot,
                                                                n_corrects/n_tot))
        with torch.no_grad():
            list_y = list()
            list_y_pred = list()
            n_corrects = 0
            n_tot = 0
            for batch in test_dataloader:
                
                y = batch.y
                list_y.append(y)
                batch = batch.to(device)
                y = y.to(device)
                out = model(batch)
                loss = loss_fn(out,y)

                tot_loss +=loss.cpu().detach().item()
                pred = out.argmax(dim=-1)
                list_y_pred.append(pred)
                n_tot+= len(pred)
                n_corrects+= (pred==y).sum()
            list_y_pred = torch.cat(list_y_pred).cpu().numpy()
            list_y = torch.cat(list_y).cpu().numpy()
            f1_score = metrics.f1_score(list_y,
                                        list_y_pred)
            roc_auc = metrics.roc_auc_score(list_y,
                                            list_y_pred)
            recall = metrics.recall_score(list_y,
                                          list_y_pred)
            test_loss  = loss/n_tot
            test_accuracy = n_corrects/n_tot
            if epoch%5==0:
                print('Test set loss : {:.3f} accuracy :{} '.format(test_loss,
                                                                   test_accuracy))
            if test_loss<= best_test_loss:
                model_desc_str = get_file_name_desc(description_exp)
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss.item(),
                'accuracy':test_accuracy,
                'roc_auc':roc_auc,
                'f1 score':f1_score,
                'recall':recall,
                'description_experiment':description_exp
                }, osp.join(save_path,
                            f'best_model_{model_desc_str}.pt')
                            )
                best_test_loss = test_loss
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Params for training models")

    # Common params
    parser.add_argument("--seed", help="random torch seed",type=int, default=1)
    parser.add_argument("--model_dir", help="Direrctory to save the models",type =str, default='models')
    parser.add_argument('--dataset_name',help='name of the dataset politifact or gossipcop',type=str,default='politifact')
    
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=200)
    parser.add_argument('--features', nargs='+', help='features to select', type = str,default='profile')
    parser.add_argument('--batch_size',help='features to select', type = int,default=8)
    parser.add_argument('--device',help='device for training model',type=str,default='cpu')
    parser.add_argument('--model',help='class of model original or deepset',type=str,default = 'original')
    args  = parser.parse_args()

    torch.manual_seed(args.seed)
    if len(args.features)==1:
        args.features=args.features[0]
    print(args.features)
    dic_models = {'original':Original_model,
                  'deepset':Deepset}

    train_dataset = UPFD('.',
                         args.dataset_name,
                         args.features,
                         'train')
    test_dataset = UPFD('.',
                        args.dataset_name,
                        args.features,
                        'test')
    model = dic_models[args.model](train_dataset)
    device = 'cpu'
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size)
    description_exp = {'dataset_name':args.dataset_name,
                       'features':args.features,
                       'batch_size':args.batch_size,
                       'model':args.model}
    train(model=model,
          train_dataloader=train_dataloader,
          test_dataloader=train_dataloader,
          device=args.device,
          save_path=args.model_dir,
          N_epochs=args.epochs,
          description_exp = description_exp
          )

