import torch
import os
import glob

files = glob.glob("/kwkj-k8s/davinci/LJH/daVinci-MagiHuman2/dataset/seedance_vertical/latents/audio/*.pt")
if not files:
    print("No audio latents found")
else:
    for f in files[:5]:
        t = torch.load(f, map_location="cpu")
        print(f"{os.path.basename(f)}: shape={t.shape}, max={t.max().item():.4f}, min={t.min().item():.4f}, rms={torch.sqrt(torch.mean(t**2)).item():.4f}")
