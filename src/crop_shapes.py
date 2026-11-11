import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

m=pd.read_csv(r"data/manifest_folds.csv")
sizes=[Image.open(p).size for p in m["path"]]
m["w"]=[s[0] for s in sizes]
m["h"]=[s[1] for s in sizes]
m["aspect"]=m["w"]/m["h"]
m["side"]=(m["w"]*m["h"])**0.5


summary = m[["w", "h", "aspect"]].describe().round(1)
by_class = m.groupby("class")["aspect"].median().sort_values().round(2)
print(summary.to_string())
print(by_class.to_string())
with open("results/crop_shapes.txt", "w") as f:
    f.write(summary.to_string() + "\n\n" + by_class.to_string())

order = m.groupby("class")["aspect"].median().sort_values().index.tolist()
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].hist(m["side"], bins=50, color="steelblue")
ax[0].set_xlabel("Crop size (square root of width x height, pixels)")
ax[0].set_ylabel("Number of crops")
ax[1].boxplot([m.loc[m["class"] == c, "aspect"] for c in order], orientation="horizontal", showfliers=False)
ax[1].set_yticks(range(1, len(order) + 1))
ax[1].set_yticklabels(order)
ax[1].set_xscale("log")
ax[1].axvline(1, color="red", linestyle="--")
ax[1].set_xlabel("Aspect ratio, width / height (log scale; 1 = square)")
plt.tight_layout()
plt.savefig("results/crop_shapes.png", dpi=200)