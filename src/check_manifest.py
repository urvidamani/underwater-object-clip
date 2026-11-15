import pandas as pd

a = pd.read_csv("data/manifest_folds.csv")
b = pd.read_csv("data/manifest.csv")
print(len(a), len(b), list(a["path"]) == list(b["path"]), list(a["class"]) == list(b["class"]))