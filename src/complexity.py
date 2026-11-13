import torch, open_clip
from torchvision import models

def n(m):
    return sum(p.numel() for p in m.parameters())

resnet=models.resnet18(num_classes=20)
clip,_,_=open_clip.create_model_and_transforms("ViT-B-32-quickgelu",pretrained="openai")

print("Resnet18, 20 classes: ",n(resnet))
print("CLIP image encoder: ",n(clip.visual))
print("CLIP whole model (image+text): ",n(clip))
print("Linear layer:",512*20+20)
print("Adapter: ",512*128+128+128*512+512+20*512)
