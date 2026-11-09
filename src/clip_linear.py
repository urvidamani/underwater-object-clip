import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score,f1_score,classification_report

manifest=r"data/manifest_folds.csv"
epochs,learning_rate,bs=100,1e-3,256
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

scores,outs=[],[]
for k in sorted(set(folds)):
    tr,te=folds!=k,folds==k
    xtr,ytr=x[tr].to(dev),y[tr].to(dev)
    xte=x[te].to(dev)
    model=nn.Linear(512,len(classes)).to(dev)
    counts=torch.bincount(ytr,minlength=len(classes)).float()
    w=counts**-0.5
    loss_fn=nn.CrossEntropyLoss(weight=(w/w.mean()).to(dev))
    opt=torch.optim.AdamW(model.parameters(),lr=learning_rate)
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
    print(f"==fold {k}: accuracy {acc:.3f} macro-f1 {f1:.3f}",flush=True)
    scores.append((acc,f1))
    out=m.loc[te,["path","class","fold"]].copy()
    out["pred"]=[classes[i] for i in p]
    outs.append(out)

s=np.array(scores)
print(f"\n Accuracy {s[:,0].mean():.3f}+-{s[:,0].std():.3f}")
print(f"\n Macro-F1 {s[:,1].mean():.3f}+-{s[:,1].std():.3f}")
all_p=pd.concat(outs)
all_p.to_csv("results/clip_linear_predictions.csv",index=False)
rep=classification_report(all_p["class"],all_p["pred"],digits=3)
print(rep)
open("results/clip_linear_report.txt","w").write(rep)