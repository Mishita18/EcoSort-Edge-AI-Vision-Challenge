import argparse, os, copy, json, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import EcoSortNet

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]

def loaders(data_dir,batch_size):
    train_tf=transforms.Compose([
        transforms.Resize((96,96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=.12,contrast=.12,saturation=.10),
        transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    val_tf=transforms.Compose([
        transforms.Resize((96,96)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    train=datasets.ImageFolder(os.path.join(data_dir,"train"),transform=train_tf)
    val=datasets.ImageFolder(os.path.join(data_dir,"val"),transform=val_tf)
    if train.classes != CLASSES or val.classes != CLASSES:
        raise ValueError(f"Class order mismatch. train={train.classes}, val={val.classes}, expected={CLASSES}")
    return (DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=0),
            DataLoader(val,batch_size=batch_size,shuffle=False,num_workers=0))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir",default="data/public")
    p.add_argument("--epochs",type=int,default=25)
    p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--output_dir",default="outputs")
    a=p.parse_args()
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader,val_loader=loaders(a.data_dir,a.batch_size)
    model=EcoSortNet().to(device)
    print("Parameters:",sum(x.numel() for x in model.parameters()),"device:",device)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=2)
    os.makedirs(a.output_dir,exist_ok=True)
    best_acc=-1; history=[]
    for epoch in range(1,a.epochs+1):
        model.train(); train_correct=train_total=0; train_loss=0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad(); logits=model(images); loss=criterion(logits,labels)
            loss.backward(); optimizer.step()
            train_loss += loss.item()*len(labels)
            train_correct += (logits.argmax(1)==labels).sum().item(); train_total += len(labels)
        model.eval(); val_correct=val_total=0; val_loss=0
        with torch.no_grad():
            for images,labels in val_loader:
                images,labels=images.to(device),labels.to(device)
                logits=model(images); loss=criterion(logits,labels)
                val_loss += loss.item()*len(labels)
                val_correct += (logits.argmax(1)==labels).sum().item(); val_total += len(labels)
        ta=train_correct/train_total; va=val_correct/val_total
        scheduler.step(va)
        row={"epoch":epoch,"train_accuracy":ta,"val_accuracy":va,"train_loss":train_loss/train_total,"val_loss":val_loss/val_total}
        history.append(row)
        print(f"Epoch {epoch:02d}/{a.epochs}: train={ta*100:.1f}% val={va*100:.1f}%")
        if va>best_acc:
            best_acc=va
            torch.save({"model_state_dict":model.cpu().state_dict(),"classes":CLASSES,"validation_accuracy":best_acc},os.path.join(a.output_dir,"best_model.pth"))
            model.to(device)
    with open(os.path.join(a.output_dir,"training_history.json"),"w") as f: json.dump(history,f,indent=2)
    print(f"Best validation accuracy: {best_acc*100:.2f}%")

if __name__=="__main__": main()
