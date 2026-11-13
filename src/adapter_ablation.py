import pandas as pd
import numpy as np
import torch
import open_clip
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,f1_score

data=r"data/manifest_folds.csv"
epochs,learning_rate,bs=100,1e-3,256
hidden,alpha,scale=128,0.2,100.0
prompt="an underwater photo of a {}"
dev="cuda"

m=pd.read_csv(data)
feats=torch.from_numpy(np.load("results/clip_img_feats.npy").copy())
assert len(m)==len(feats)
core=(m["fold"]>=0).values
m=m[core].reset_index(drop=True)
x=feats[torch.from_numpy(core)]
classes=sorted(m["class"].unique())
y=torch.tensor(m["class"].map({c: i for i,c in enumerate(classes)}).values)
folds=m["fold"].values

clip,_,_=open_clip.create_model_and_transforms("ViT-B-32-quickgelu",pretrained="openai")
tok=open_clip.get_tokenizer("ViT-B-32-quickgelu")
clip.to(dev).eval()
with torch.no_grad():
    T=F.normalize(clip.encode_text(tok([prompt.format(c.lower())for c in classes]).to(dev)).float(),dim=-1)
del clip
torch.cuda.empty_cache()

class Head(nn.Module):
    def __init__(self,text_emb,use_adapter,text_init):
        super().__init__()
        d=text_emb.shape[1]
        self.use_adapter=use_adapter
        if use_adapter:
            self.adapter=nn.Sequential(nn.Linear(d,hidden),nn.ReLU(),nn.Linear(hidden,d))
        w=text_emb.clone() if text_init else F.normalize(torch.rand_like(text_emb),dim=-1)
        self.w=nn.Parameter(w)
    def forward(self,f):
        if self.use_adapter:
            f=alpha*self.adapter(f)+(1-alpha)*f
        f=F.normalize(f,dim=-1)
        return scale*f@F.normalize(self.w,dim=-1).T 
    
variants={"cosine classifier, random init":(False,False),
          "cosine classifier, text init":(False, True),
          "adapter + random init":(True,False),
          "adapter + text init (full)":(True,True)}
rows=[]
for name,(use_ad,txt) in variants.items():
    scores=[]
    for k in sorted(set(folds)):
        torch.manual_seed(42+k)
        tr,te=folds!=k,folds==k
        xtr,ytr,xte=x[tr].to(dev),y[tr].to(dev),x[te].to(dev)
        model=Head(T,use_ad,txt).to(dev)
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
        scores.append((accuracy_score(y[te].numpy(),p),f1_score(y[te].numpy(),p,average="macro"))) 
    s=np.array(scores)
    print(f"{name:34s} acc {s[:,0].mean():.3f} +- {s[:,0].std():.3f} | F1 {s[:,1].mean():.3f} +- {s[:,1].std():.3f}", flush=True)
    rows.append([name, s[:,0].mean(), s[:,0].std(), s[:,1].mean(), s[:,1].std()])  
pd.DataFrame(rows,columns=["variant","acc","acc_std","f1","f1_std"]).to_csv("results/adapter_ablation.csv",index=False)