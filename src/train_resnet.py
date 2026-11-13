from typing import Any

import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score,f1_score,classification_report
from torchvision import models,transforms

data=r"data/manifest_folds.csv"
OUT=Path("results")
OUT.mkdir(exist_ok=True)
epochs,bs,Learning_rate=5,64,3e-4
torch.manual_seed(42)

norm=transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
train_tf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(0.7,1.0)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),norm])
test_tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),norm])


class CropDS(Dataset):
    def __init__(self,df,tf):
        self.p,self.y,self.tf=df["path"].tolist(),df["y"].tolist(),tf
    def __len__(self):
        return len(self.p)
    def __getitem__(self, index):
        return self.tf(Image.open(self.p[index]).convert("RGB")),self.y[index]

def loader(df,tf,shuffle):
    return DataLoader(CropDS(df,tf),batch_size=bs,shuffle=shuffle,num_workers=4,pin_memory=True)

def main():
    m=pd.read_csv(data)
    m=m[m["fold"]>=0].reset_index(drop=True)
    classes=sorted(m["class"].unique())
    m["y"]=m["class"].map({c:i for i,c in enumerate(classes)})
    print(len(classes),"classes: ",classes)
    dev="cuda"
    preds_all,scores=[],[]
    
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for k in sorted(m["fold"].unique()):
        tr,te=m[m["fold"]!=k],m[m["fold"]==k]
        model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc=nn.Linear(512,len(classes))
        model.to(dev)
        
        counts=tr["y"].value_counts().sort_index().values
        w=counts**-0.5
        w=torch.tensor(w/w.mean(),dtype=torch.float32,device=dev)
        loss_fn=nn.CrossEntropyLoss(weight=w)
        opt=torch.optim.AdamW(model.parameters(),lr=Learning_rate)
        scaler=torch.amp.GradScaler("cuda")
        tl=loader(tr,train_tf,True)
        
        for ep in range(epochs):
            model.train()
            tot=0
            for x,y in tl:
                x,y=x.to(dev),y.to(dev)
                opt.zero_grad()
                with torch.amp.autocast("cuda"):
                    loss=loss_fn(model(x),y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot+=loss.item()
            print(f"fold {k} epoch {ep+1}/{epochs} loss {tot/len(tl):.3f}",flush=True)
        
        model.eval()
        p=[]
        with torch.no_grad():
            for x,_ in loader(te,test_tf,False):
                p+=model(x.to(dev)).argmax(1).cpu().tolist()
        acc=accuracy_score(te["y"],p)
        f1=f1_score(te["y"],p,average="macro")
        print(f"== fold {k}: accuracy {acc:.3f}  macro-F1 {f1:.3f}", flush=True)
        scores.append((acc,f1))
        out=te[["path","class","fold"]].copy()
        out["pred"]=[classes[i] for i in p]
        preds_all.append(out)
    
    print(f"Training + eval time (3 folds): {time.time() - t0:.1f} s")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
    s=np.array(scores)
    print(f"\n Accuracy {s[:,0].mean():.3f} +-{s[:,0].std():.3f}")
    print(f"\n Macro-F1 {s[:,1].mean():.3f} +-{s[:,1].std():.3f}")
    allp=pd.concat(preds_all)
    allp.to_csv(OUT/"resnet18_predictions.csv", index=False)
    rep=classification_report(allp["class"],allp["pred"],digits=3)
    print(rep)
    (OUT/"resnet18_report.txt").write_text(rep)
    

if __name__=="__main__":
    main()
        