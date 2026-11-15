import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from sklearn.metrics import accuracy_score, f1_score

manifest = "data/manifest_folds.csv"
epochs, learning_rate, bs = 100, 1e-3, 256
hidden, alpha, scale = 128, 0.2, 100.0
prompt = "an underwater photo of a {}"
seeds = [0, 1, 2, 3, 4]
dev = "cuda"

# Step 1: load the crop list and the saved (padded) CLIP features
m = pd.read_csv(manifest)
feats = torch.from_numpy(np.load("results/clip_img_feats.npy").copy())
assert len(m) == len(feats)
core = (m["fold"] >= 0).values
m = m[core].reset_index(drop=True)
x = feats[torch.from_numpy(core)]
classes = sorted(m["class"].unique())
y = torch.tensor(m["class"].map({c: i for i, c in enumerate(classes)}).values)
folds = m["fold"].values

# Step 2: encode the class prompts once (used to start the adapter's classifier)
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")
clip.to(dev).eval()
with torch.no_grad():
    T = clip.encode_text(tok([prompt.format(c.lower()) for c in classes]).to(dev))
    T = F.normalize(T.float(), dim=-1)
del clip
torch.cuda.empty_cache()


class TextInitAdapter(nn.Module):
    def __init__(self, text_emb):
        super().__init__()
        d = text_emb.shape[1]
        self.adapter = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.w = nn.Parameter(text_emb.clone())

    def forward(self, f):
        f = alpha * self.adapter(f) + (1 - alpha) * f
        f = F.normalize(f, dim=-1)
        return scale * f @ F.normalize(self.w, dim=-1).T


def run(kind, seed):
    """Train one model type on each fold with one seed; return mean accuracy and macro-F1 over folds."""
    torch.manual_seed(seed)
    res = []
    for k in sorted(set(folds)):
        tr, te = folds != k, folds == k
        xtr, ytr, xte = x[tr].to(dev), y[tr].to(dev), x[te].to(dev)
        model = (TextInitAdapter(T) if kind == "adapter" else nn.Linear(512, len(classes))).to(dev)
        counts = torch.bincount(ytr, minlength=len(classes)).float()
        w = counts ** -0.5
        loss_fn = nn.CrossEntropyLoss(weight=(w / w.mean()).to(dev))
        wd = 1e-4 if kind == "adapter" else 1e-2          
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
        for ep in range(epochs):
            perm = torch.randperm(len(xtr), device=dev)
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss_fn(model(xtr[b]), ytr[b]).backward()
                opt.step()
        with torch.no_grad():
            p = model(xte).argmax(1).cpu().numpy()
        res.append((accuracy_score(y[te].numpy(), p), f1_score(y[te].numpy(), p, average="macro")))
    return np.array(res).mean(axis=0)


rows = []
t0 = time.time()
for kind in ["linear", "adapter"]:
    per_seed = []
    for sd in seeds:
        acc, f1 = run(kind, sd)
        print(f"{kind:8s} seed {sd}: accuracy {acc:.3f}  macro-F1 {f1:.3f}", flush=True)
        per_seed.append((acc, f1))
        rows.append([kind, sd, acc, f1])
    s = np.array(per_seed)
    print(f"{kind:8s} over 5 seeds: accuracy {s[:,0].mean():.3f} +- {s[:,0].std():.3f} | "
          f"macro-F1 {s[:,1].mean():.3f} +- {s[:,1].std():.3f}\n", flush=True)

pd.DataFrame(rows, columns=["model", "seed", "acc", "f1"]).to_csv("results/seeds_5.csv", index=False)
print(f"Total time: {time.time() - t0:.0f} s")