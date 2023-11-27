import torch.nn as nn
from torch import optim
from models import Original_model
from dataset import UPFD
from torch_geometric.loader import DataLoader

def train(model,train_dataloader,test_dataloader,N_epochs,device='cuda:0'):
    loss_fn = nn.CrossEntropyLoss()
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    for epoch in range(N_epochs):
        tot_loss = 0
        n_corrects = 0
        n_tot = 0
        for batch in train_dataloader:
            print(batch)
            y = batch.y

            optimizer.zero_grad()
            batch = batch.to(device)
            y = y.to(device)
            out = model(batch)
            print(out.shape)
            print(y.shape)
            loss = loss_fn(out,y)
            loss.backward()
            optimizer.step()
            tot_loss +=loss.cpu().detach().item()
            pred = out.argmax(dim=-1)
            n_tot+= len(pred)
            n_corrects+= (pred==y).sum()

        print('Epoch {} loss : {:.3f} accuracy :{} '.format(epoch,
                                                            loss/n_tot,
                                                            n_corrects/n_tot))
if __name__=='__main__':
    dataset = UPFD('.','politifact','profile')
    model = Original_model(dataset)
    device = 'cpu'
    train_dataloader = DataLoader(dataset,batch_size=3)
    train(model=model,
          train_dataloader=train_dataloader,
          test_dataloader=train_dataloader,
          device=device,
          N_epochs=1)

