from pathlib import Path
from PIL import Image
import csv
from collections import Counter

Root=Path("C:/Users/urvi_/OneDrive - Adelaide University/Desktop/Sem2/CV/underwater-object-clip/data/YOLO")
Crop_images=Path("C:/Users/urvi_/OneDrive - Adelaide University/Desktop/Sem2/CV/underwater-object-clip/data")
Padding=0.10
Minimum_size=32

names=[l.strip().split(":",1)[-1].strip() for l in open(Root/"obj.names") if l.strip()]
print(len(names),"classes: ",names)

rows=[]
for split in ["train","val","test"]:
    for lab in sorted((Root/"labels"/split).glob("*.txt")):
        img_path=Root/"images"/split/(lab.stem+".jpg")
        if not img_path.exists():
            continue
        img=Image.open(img_path).convert("RGB")
        w,h=img.size
        for i,line in enumerate(open(lab)):
            p=line.split()
            if len(p)<5:
                continue
            cls=int(p[0])
            if names[cls]=="Unknown Instance":
                continue
            v=list(map(float,p[1:]))
            if len(v)==4:
                cx,cy,w,h=v
                x0,x1,y0,y1=cx-w/22,cx+w/2,cy-h/2,cy+h/2
            else:
                xs,ys=v[0::2],v[1::2]
                x0,x1,y0,y1=min(xs),max(xs),min(ys),max(ys)
            bw,bh=(x1-x0)*w,(y1-y0)*h
            if bw<Minimum_size or bh<Minimum_size:
                continue
            px,py=bw*Padding,bh*Padding
            box=(max(0,x0*w-px),max(0,y0*h-py),
                min(w,x1*w+px),min(h,y1*h+py))
            crop=img.crop(tuple(int(b) for b in box))
            d=Crop_images/split/names[cls]
            d.mkdir(parents=True,exist_ok=True)
            fn=d/f"{lab.stem}_{i}.jpg" 
            crop.save(fn,quality=95)
            rows.append([split,names[cls],str(fn),lab.stem.rsplit("_",1)[0]])

Crop_images.mkdir(parents=True,exist_ok=True)
with open(Crop_images/"manifest.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["split","class","path","source"])
    w.writerows(rows)

print("\n Total crops: ",len(rows))
for s in ["train","val","test"]:
    print(s,sum(1 for r in rows if r[0]==s))
print("\n Per class: ")
for c,n in Counter(r[1] for r in rows).most_common():
    print(f" {c}: {n}")
    
    