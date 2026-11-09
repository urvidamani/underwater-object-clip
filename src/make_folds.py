import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

m=pd.read_csv(r"data\manifest.csv")

m['frame']=m['path'].str.extract(r"_(\d+)_\d+\.jpg$")[0].astype(int)
rare=["Pipe","Plastic Bag"]
core=m[~m["class"].isin(rare)].copy()
core["group"]=core["source"]+"_"+(core["frame"]//25).astype(str)

sgkf=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42)
m["fold"]=-1
for k,(_,te) in enumerate(sgkf.split(core,core["class"],groups=core["group"])):
    m.loc[core.index[te],"fold"]=k

m.to_csv(r"data/manifest_folds.csv",index=False)
print(pd.crosstab(m["class"],m["fold"]))

# print(pd.crosstab(m["class"], m["fold"]).loc[["Pipe", "Plastic Bag", "Case", "Dive Weight", "Aqua", "LoCo"]])