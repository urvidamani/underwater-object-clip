from pathlib import Path
Root=Path("C:/Users/urvi_/OneDrive - Adelaide University/Desktop/Sem2/CV/underwater-object-clip/data/YOLO")
n = 0
for lab in (Root / "labels").rglob("*.txt"):
    for line in open(lab):
        if line.split()[:1] == ["3"]:
            n += 1
print("Metal Rod objects in labels:", n)

import pandas as pd
m = pd.read_csv(r"C:/Users/urvi_/OneDrive - Adelaide University/Desktop/Sem2/CV/underwater-object-clip/data/manifest.csv")
print(pd.crosstab(m["class"], m["split"]))
print(pd.crosstab(m["source"], m["split"]))