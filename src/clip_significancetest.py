import pandas as pd
from scipy.stats import binomtest

files={"resnet18":"results/resnet18_predictions.csv",
       "resnet18_padded":"results/resnet18_padded_predictions.csv",
       "clip_linear":"results/clip_linear_predictions.csv",
       "clip_adapter":"results/clip_adapter_predictions.csv"}

preds={k:pd.read_csv(v) for k,v in files.items()}

def compare(a,b):
    d=preds[a].merge(preds[b],on=["path","class"],suffixes=("_a","_b"))
    assert len(d)==len(preds[a])==len(preds[b])
    ca,cb=d["pred_a"]==d["class"],d["pred_b"]==d["class"]
    only_a,only_b=int((ca & ~cb).sum()),int((~ca &cb).sum())
    p=binomtest(only_a,only_a+only_b,0.5).pvalue
    print(f"{a} vs {b}:only {a} right ={only_a},only {b} right ={only_b},p={p:.4g}")
    
compare("resnet18","clip_adapter")
compare("clip_linear","clip_adapter")
compare("resnet18","clip_linear")
compare("resnet18_padded","clip_adapter")
compare("resnet18_padded","clip_linear")
compare("resnet18","resnet18_padded")