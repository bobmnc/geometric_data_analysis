import torch.nn as nn
from torch import optim
from models import Original_model
from dataset import UPFD
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch

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
    train_dataset = UPFD('.',
                         'politifact',
                         'profile')
    test_dataset = UPFD('.',
                        'politifact',
                        'profile',
                        'test')
    model = Original_model(train_dataset)
    device = 'cpu'
    train_dataloader = DataLoader(train_dataset,batch_size=4)
    test_dataloader = DataLoader(test_dataset,batch_size=4)
    train(model=model,
          train_dataloader=train_dataloader,
          test_dataloader=train_dataloader,
          device=device,
          N_epochs=200)

