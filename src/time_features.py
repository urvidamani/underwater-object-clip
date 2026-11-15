import time, pandas as pd, torch, open_clip
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader

def pad_square(im):
    s = max(im.size)
    return ImageOps.pad(im, (s, s), color=(0, 0, 0))

class DS(Dataset):
    def __init__(self, paths, tf): 
        self.p, self.tf = paths, tf
    def __len__(self): 
        return len(self.p)
    def __getitem__(self, i): 
        return self.tf(pad_square(Image.open(self.p[i]).convert("RGB")))

def main():
    m = pd.read_csv("data/manifest_folds.csv")
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    model.to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for x in DataLoader(DS(m["path"].tolist(), pre), batch_size=128, num_workers=4):
            model.encode_image(x.to("cuda"))
    torch.cuda.synchronize()
    print(f"Feature extraction: {time.time() - t0:.1f} s for {len(m)} crops (includes loading images)")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")

if __name__ == "__main__":
    main()