import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

m=pd.read_csv(r"data/manifest_folds.csv")

counts=m["class"].value_counts().sort_values()
fig,ax=plt.subplots(figsize=(9,6))
counts.plot.barh(ax=ax,color="steelblue")
ax.set_xscale("log")
ax.set_xlabel("Number of crops (log scale)")

for i,v in enumerate(counts):
    ax.text(v,i,f"{v}",va="center",fontsize=8)

plt.tight_layout()
plt.savefig("results/class_counts.png",dpi=200)


classes=sorted(m["class"].unique())
n=5
fig,axes=plt.subplots(len(classes),n,figsize=(n*1.4,len(classes)*1.3))
for r,c in enumerate(classes):
    sub=m[m["class"]==c]
    s=sub.sample(min(n,len(sub)),random_state=0)
    for j in range(n):
        ax=axes[r,j]
        ax.axis("off")
        if j<len(s):
            ax.imshow(Image.open(s.iloc[j]["path"]))
        if j==0:
            ax.set_title(c,fontsize=7,loc="left")

plt.tight_layout()
plt.savefig("results/sample_crops.png",dpi=150)
print("saved")