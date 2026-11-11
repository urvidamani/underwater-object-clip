import time
from typing import Any
import pandas as pd
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import open_clip
from sklearn.metrics import accuracy_score,f1_score,classification_report

manifest=r"data/manifest_folds.csv"
epochs,learning_rate,bs=100,1e-3,256
hidden,alpha,scale=128,0.2,100.0
prompt="an underwater photo of a {}"
torch.manual_seed(42)
dev="cuda"

m=pd.read_csv(manifest)
feats=torch.from_numpy(np.load("results/clip_img_feats.npy").copy())
assert len(m)==len(feats)
core=(m["fold"]>=0).values
m=m[core].reset_index(drop=True)
x=feats[torch.from_numpy(core)]
classes=sorted(m["class"].unique())
y=torch.tensor(m["class"].map({c:i for i,c in enumerate(classes)}).values)
folds=m["fold"].values

clip,_,_=open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
tok=open_clip.get_tokenizer("ViT-B-32-quickgelu")

clip.to(dev).eval()
with torch.no_grad():
    T=clip.encode_text(tok([prompt.format(c.lower())for c in classes]).to(dev))
    T=F.normalize(T.float(),dim=-1)

del clip
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


class TextInitAdapter(nn.Module):
    def __init__(self,text_emb):
        super().__init__()
        d=text_emb.shape[1]
        self.adapter=nn.Sequential(nn.Linear(d,hidden),nn.ReLU(),nn.Linear(hidden,d))
        self.w=nn.Parameter(text_emb.clone())
    def forward(self,f):
        f=alpha*self.adapter(f)+(1-alpha)*f
        f=F.normalize(f,dim=-1)
        return scale*f@F.normalize(self.w,dim=-1).T

scores,outs=[],[]
t0=time.time()
for k in sorted(set(folds)):
    tr,te=folds!=k,folds==k
    xtr,ytr=x[tr].to(dev),y[tr].to(dev)
    xte=x[te].to(dev)
    model=TextInitAdapter(T).to(dev)
    counts=torch.bincount(ytr,minlength=len(classes)).float()
    w=counts**-0.5
    loss_fn=nn.CrossEntropyLoss(weight=(w/w.mean()).to(dev))
    opt=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    for ep in range(epochs):
        perm=torch.randperm(len(xtr),device=dev)
        for i in range(0,len(perm),bs):
            b=perm[i:i+bs]
            opt.zero_grad()
            loss_fn(model(xtr[b]),ytr[b]).backward()
            opt.step()
    with torch.no_grad():
        p=model(xte).argmax(1).cpu().numpy()
    acc=accuracy_score(y[te].numpy(),p)
    f1=f1_score(y[te].numpy(),p,average="macro")
    print(f"== fold {k}: accuracy {acc:.3f} macro-f1 {f1:.3f}",flush=True)
    scores.append((acc,f1))
    o=m.loc[te,["path","class","fold"]].copy()
    o["pred"]=[classes[i] for i in p]
    outs.append(o)
    
s=np.array(scores)
print(f"\n Accuracy {s[:,0].mean():.3f}+-{s[:,0].std():.3f}")
print(f"\n Macro-f1 {s[:,1].mean():.3f}+-{s[:,1].std():.3f}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Training time (3 folds): {time.time()-t0:.1f}s")
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e6:.0f}MB")
allp=pd.concat(outs)
allp.to_csv("results/clip_adapter_predictions.csv",index=False)
rep=classification_report(allp["class"],allp["pred"],digits=3)
print(rep)
open("results/clip_adapter_report.txt","w").write(rep)
    