import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

p=pd.read_csv("results/resnet18_predictions.csv")
classes=sorted(p["class"].unique())
cm=confusion_matrix(p["class"],p["pred"],labels=classes)
cmn=cm/cm.sum(axis=1,keepdims=True)

fig,ax=plt.subplots(figsize=(10,9))
im=ax.imshow(cmn,cmap="Blues")
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes,rotation=90)
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes)
ax.set_xlabel("predicted")
ax.set_ylabel("True")
fig.colorbar(im,label="Fraction of true class")
plt.tight_layout()

pairs=[(cm[i,j],classes[i],classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i!=j]
for n,a,b in sorted(pairs,reverse=True)[:10]:
    print(f"{a}->{b}:{n}")
    
plt.show()