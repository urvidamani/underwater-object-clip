import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

epochs, leanring_rate, bs = 100, 1e-3, 256
torch.manual_seed(42)
dev = "cuda"
m = pd.read_csv("data/manifest_folds.csv")
feats = torch.from_numpy(np.load("results/clip_img_feats.npy").copy())
core = (m["fold"] >= 0).values
m = m[core].reset_index(drop=True)
X = feats[torch.from_numpy(core)]
classes = sorted(m["class"].unique())
y = torch.tensor(m["class"].map({c: i for i, c in enumerate(classes)}).values)
folds = m["fold"].values

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for k in sorted(set(folds)):
    tr = folds != k
    Xtr, ytr = X[tr].to(dev), y[tr].to(dev)
    model = nn.Linear(512, len(classes)).to(dev)
    counts = torch.bincount(ytr, minlength=len(classes)).float()
    w = counts ** -0.5
    loss_fn = nn.CrossEntropyLoss(weight=(w / w.mean()).to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=leanring_rate)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(model(Xtr[b]), ytr[b]).backward()
            opt.step()
torch.cuda.synchronize()
print(f"Linear layer training (3 folds): {time.time() - t0:.1f} s")
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")