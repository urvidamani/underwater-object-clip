import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

name= sys.argv[1] if len(sys.argv)>1 else "clip_adapter"

p=pd.read_csv(f"results/{name}_predictions.csv")
classes=sorted(p["class"].unique())
cm=confusion_matrix(p["class"],p["pred"],labels=classes)
cmn=cm/cm.sum(axis=1,keepdims=True)

fig,ax=plt.subplots(figsize=(10,9))
im=ax.imshow(cmn,cmap="Blues")
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes,rotation=90)
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
fig.colorbar(im,label="Fraction of true class")
plt.tight_layout()

pairs=[(cm[i,j],classes[i],classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i!=j]
for n,a,b in sorted(pairs,reverse=True)[:10]:
    print(f"{a}->{b}:{n}")

plt.savefig(f"results/{name}_confusion.png", dpi=200)
plt.show()