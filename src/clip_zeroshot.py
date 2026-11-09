from typing import Any

import pandas as pd
import numpy as np
import torch
import open_clip
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import accuracy_score,f1_score,classification_report

manifest=r"data/manifest_folds.csv"
rare=["Pipe","Plastic Bag"]
templates={"Plain":"a photo of a {}","Underwater":"an underwater photo of a {}"}
dev="cuda"

class DS(Dataset):
    def __init__(self,paths,tf):
        self.p,self.tf=paths,tf
    def __len__(self):
        return len(self.p)
    def __getitem__(self, index):
        return self.tf(Image.open(self.p[index]).convert("RGB"))
    
def main():
    m=pd.read_csv(manifest)
    classes=sorted(m["class"].unique())
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    model.to(dev).eval()
    tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    
    feats=[]
    with torch.no_grad():
        for x in DataLoader(DS(m["path"].tolist(),pre),batch_size=128,num_workers=4):
            f=model.encode_image(x.to(dev))
            feats.append((f/f.norm(dim=-1,keepdim=True)).cpu())
    feats=torch.cat(feats)
    np.save("results/clip_img_feats.npy",feats.numpy())
    
    core=(m["fold"]>=0).values
    index20=[i for i,c in enumerate(classes) if c not in rare]
    best=None
    for name,t in templates.items():
        with torch.no_grad():
            T=model.encode_text(tok([t.format(c.lower()) for c in classes]).to(dev))
            T=(T/T.norm(dim=1,keepdim=True)).cpu()
        logits=feats@T.T
        
        sub=logits[core][:,index20]
        pred=[classes[index20[j]]for j in sub.argmax(1).tolist()]
        true=m.loc[core,"class"].tolist()
        acc,f1=accuracy_score(true,pred),f1_score(true,pred,average="macro")
        
        pred22=[classes[j] for j in logits.argmax(1).tolist()]
        acc22=accuracy_score(m["class"],pred22)
        f122=f1_score(m["class"],pred22,average="macro")
        print(f"{name:10s} 20-way:acc {acc:.3f} F1 {f1:.3f} | 22-way: acc{acc22:.3f} F1 {f122:.3f}")
        if best is None or f1>best[0]:
            best=(f1,name,pred,true,pred22)
    
    _,name,pred,true,pred22=best
    print("\n Best template: ",name)
    rep=classification_report(true,pred,digits=3)
    print(rep)
    open("results/clip_zeroshot_report.txt","w").write(rep)
    out=m.copy()
    out["pred22"]=pred22
    out.to_csv("results/clip_zeroshot_predictions.csv",index=False)

if __name__=="__main__":
    main()
        