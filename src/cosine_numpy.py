"""
Cosine Claasifier on frozen CLIP features written from scratch in NumPy. 
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
manifest="data/manifest_folds.csv"
features="results/clip_img_feats.npy"
epochs,learning_rate,bs=100,1e-3,256
weight_decay,scale=1e-4,100.0

def forward(f,w,s):
    norms=np.linalg.norm(w,axis=1,keepdims=True)
    u=w/norms
    return s*f@u.T,u,norms
def loss_and_grad(f,y,w,s,class_w):
    n=len(y)
    Z,U,norms=forward(f,w, s)
    Z=Z-Z.max(axis=1,keepdims=True)
    P=np.exp(Z)
    P/=P.sum(axis=1,keepdims=True)
    wy=class_w[y]
    loss=-(wy * np.log(P[np.arange(n), y] + 1e-12)).sum() / wy.sum()
    dZ=P.copy()
    dZ[np.arange(n), y]-=1.0
    dZ*=(wy / wy.sum())[:, None]
    dU=s * dZ.T @ f
    dW=(dU -(dU*U).sum(axis=1, keepdims=True)*U)/norms
    return loss, dW

def gradient_check(seed=0):
    rng=np.random.default_rng(seed)
    f=rng.standard_normal((32, 16))
    f/=np.linalg.norm(f, axis=1, keepdims=True)
    y=rng.integers(0, 5, size=32)
    w=rng.standard_normal((5, 16))
    cw=rng.uniform(0.5, 2.0, size=5)
    _, g = loss_and_grad(f, y, w, 10.0, cw)
    worst,eps = 0.0, 1e-6
    for _ in range(20):
        i,j=rng.integers(0, 5), rng.integers(0, 16)
        Wp,Wm =w.copy(),w.copy()
        Wp[i,j]+=eps
        Wm[i,j]-=eps
        num=(loss_and_grad(f, y, Wp, 10.0, cw)[0] - loss_and_grad(f, y, Wm, 10.0, cw)[0]) / (2 * eps)
        worst=max(worst, abs(num - g[i, j]) / (abs(num) + abs(g[i, j]) + 1e-12))
    return worst

class Adam:
    def __init__(self, shape,lr,wd,b1=0.9,b2=0.999,eps=1e-8):
        self.m,self.v=np.zeros(shape),np.zeros(shape)
        self.lr,self.wd,self.b1,self.b2,self.eps,self.t=lr,wd,b1,b2,eps,0
 
    def step(self,w,g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        w=w * (1 - self.lr * self.wd)              
        return w- self.lr * mhat / (np.sqrt(vhat) + self.eps)
    
def main():
    err = gradient_check()
    print(f"Gradient check: worst relative error {err:.2e} (should be below 1e-5)")
    assert err < 1e-5, "hand-derived gradient does not match finite differences"
 
    m = pd.read_csv(manifest)
    feats = np.load(features).astype(np.float64)
    assert len(m) == len(feats), "manifest and features must have the same rows"
    core = (m["fold"] >= 0).values                  
    m = m[core].reset_index(drop=True)
    X = feats[core]
    classes = sorted(m["class"].unique())
    y = m["class"].map({c: i for i, c in enumerate(classes)}).values
    folds = m["fold"].values
 
    scores, preds = [], np.empty(len(y), dtype=int)
    for k in sorted(set(folds)):
        tr, te = folds != k, folds == k
        Xtr, ytr = X[tr], y[tr]
        rng = np.random.default_rng(42 + k)
        W = rng.standard_normal((len(classes), X.shape[1]))
        W /= np.linalg.norm(W, axis=1, keepdims=True)  # random unit-length class vectors
        counts = np.bincount(ytr, minlength=len(classes)).astype(float)
        class_w = counts ** -0.5
        class_w /= class_w.mean()                      # class weights proportional to count^-0.5
        opt = Adam(W.shape, learning_rate, weight_decay)
        for ep in range(epochs):
            order = rng.permutation(len(ytr))
            for i in range(0, len(order), bs):
                b = order[i:i + bs]
                _, g = loss_and_grad(Xtr[b], ytr[b], W, scale, class_w)
                W = opt.step(W, g)
        P = forward(X[te], W, scale)[0].argmax(axis=1)
        preds[te] = P
        acc, f1 = accuracy_score(y[te], P), f1_score(y[te], P, average="macro")
        print(f"== fold {k}: accuracy {acc:.3f}  macro-F1 {f1:.3f}", flush=True)
        scores.append((acc, f1))
 
    s = np.array(scores)
    print(f"\nAccuracy {s[:, 0].mean():.3f} +- {s[:, 0].std():.3f}")
    print(f"Macro-F1 {s[:, 1].mean():.3f} +- {s[:, 1].std():.3f}")
    print(f"Trainable parameters: {W.size}")
    out = m[["path", "class", "fold"]].copy()
    out["pred"] = [classes[i] for i in preds]
    out.to_csv("results/cosine_numpy_predictions.csv", index=False)
 
 
if __name__ == "__main__":
    main()