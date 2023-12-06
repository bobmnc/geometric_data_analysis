import torch.nn as nn
from torch import optim
from models import Original_model
from dataset import UPFD
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch
import argparse

def train(model,train_dataloader,test_dataloader,N_epochs,device='cuda:0'):
    loss_fn = nn.CrossEntropyLoss()
    
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
            for batch in test_dataloader:
                
                y = batch.y
                batch = batch.to(device)
                y = y.to(device)
                out = model(batch)
                loss = loss_fn(out,y)

                tot_loss +=loss.cpu().detach().item()
                pred = out.argmax(dim=-1)
                n_tot+= len(pred)
                n_corrects+= (pred==y).sum()
            if epoch%5==0:
                print('Test set loss : {:.3f} accuracy :{} '.format(loss/n_tot,
                                                                    n_corrects/n_tot))
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Params for training models")

    # Common params
    parser.add_argument("--seed", help="random torch seed",type=int, default=1)
    parser.add_argument("--model_dir", help="Direrctory to save the models",type =str, default='models')
    
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=200)
    parser.add_argument('--features', nargs='+', help='features to select', type = str,default='profile')
    parser.add_argument('--batch_size',help='features to select', type = int,default=8)
    args  = parser.parse_args()
    train_dataset = UPFD('.',
                         'politifact',
                         args.features,
                         'train')
    test_dataset = UPFD('.',
                        'politifact',
                        args.features,
                        'test')
    model = Original_model(train_dataset)
    device = 'cpu'
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size)
    train(model=model,
          train_dataloader=train_dataloader,
          test_dataloader=train_dataloader,
          device=device,
          N_epochs=args.epochs)

