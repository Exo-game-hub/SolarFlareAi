import os
import glob
import time
import json
import hashlib
import subprocess
import importlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

# Define constants
H, W = 2048, 2048
DISK_R = 950
TTA = 8
KEEP = dict(t_lo=0.35, t_hi=0.99, min_area=200, min_meanp=0.85)
D4 = [(0, False), (0, True), (2, False), (2, True), (1, False), (1, True), (3, False), (3, True)]
S8 = np.ones((3, 3), dtype=bool)

# Generate solar disk mask
YY, XX = np.ogrid[:H, :W]
DISK = ((YY - (H - 1) / 2.0) ** 2 + (XX - (W - 1) / 2.0) ** 2) <= DISK_R ** 2

def setup_environment(input_dir):
    """Installs required wheels and imports packages dynamically."""
    def find_file(name):
        hits = sorted(glob.glob(os.path.join(input_dir, "**", name), recursive=True))
        assert len(hits) == 1, f"Expected exactly one {name} under {input_dir}"
        return hits[0]

    receipt_path = find_file("solar_unet_receipt.json")
    weights_dir = os.path.dirname(receipt_path)
    test_dir = [d for d in sorted(glob.glob(os.path.join(input_dir, "**", "test_images"), recursive=True)) if os.path.isdir(d)][0]

    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        wheel_path = find_file("segmentation_models_pytorch-0.5.0-py3-none-any.whl")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", wheel_path], check=True)
        importlib.invalidate_caches()
    
    return receipt_path, weights_dir, test_dir

def load_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)

def disk_stats(img):
    v = img[DISK].astype(np.float32) / 255.0
    return v.mean(), max(float(v.std()), 1e-3)

def _tf(z, k, flip):
    if flip: z = torch.flip(z, dims=[3])
    return torch.rot90(z, k, dims=[2, 3]) if k else z

def _itf(p, k, flip):
    if k: p = torch.rot90(p, -k, dims=[2, 3])
    return torch.flip(p, dims=[3]) if flip else p

def predict(model, img_u8, mean, std, device):
    import torch
    gpu = device.type == "cuda"
    x = torch.from_numpy(np.array(img_u8, dtype=np.uint8, copy=True)).to(device).float().div_(255.0)
    z = ((x - float(mean)) / float(std))[None, None]
    acc = torch.zeros((H, W), dtype=torch.float32, device=device)
    
    for k, fl in D4[:TTA]:
        inp = _tf(z, k, fl)
        if gpu: inp = inp.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=gpu):
            out = model(inp)
        prob = torch.sigmoid(out.float())
        acc += _itf(prob, k, fl)[0, 0]
        
    acc /= TTA
    return (acc * 255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()

def components(p_u8, t_lo):
    u8_thresh = int(round(float(t_lo) * 255))
    lab, n = ndi.label(p_u8 > u8_thresh, structure=S8)
    lab_f = lab.ravel()
    nz = np.flatnonzero(lab_f)
    lv, pv = lab_f[nz], p_u8.ravel()[nz]
    area = np.bincount(lv, minlength=n + 1)[1:].astype(np.int64)
    psum = np.bincount(lv, weights=pv, minlength=n + 1)[1:]
    pmax = np.zeros(n + 1, np.uint8)
    np.maximum.at(pmax, lv, pv)
    meanp = psum / np.maximum(area, 1) / 255.0
    return lab, area, meanp, pmax[1:]

def main():
    import torch
    import segmentation_models_pytorch as smp
    from pycocotools import mask as mask_util

    input_path = "/kaggle/input" if os.path.exists("/kaggle/input") else "./input"
    output_path = "/kaggle/working" if os.path.exists("/kaggle/working") else "./working"
    os.makedirs(output_path, exist_ok=True)

    receipt_path, weights_dir, test_dir = setup_environment(input_path)
    
    with open(receipt_path) as f:
        receipt = json.load(f)
    models_meta = receipt["ensemble"]["models"]

    stems = sorted(os.path.basename(p)[:-len(".jpeg")] for p in glob.glob(os.path.join(test_dir, "*.jpeg")))
    with ThreadPoolExecutor(8) as ex:
        test_u8 = np.stack(list(ex.map(load_gray, [os.path.join(test_dir, s + ".jpeg") for s in stems])))
    stats = np.array([disk_stats(im) for im in test_u8], dtype=np.float32)

    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [torch.device("cpu")]
    total = np.zeros((len(stems), H, W), np.uint16)
    lock = threading.Lock()
    todo = list(models_meta)

    def worker(device):
        while True:
            with lock:
                if not todo: return
                m = todo.pop(0)
            model = smp.Unet(encoder_name=m["encoder"], encoder_weights=None, in_channels=1, classes=1)
            model.load_state_dict(torch.load(os.path.join(weights_dir, m["file"]), map_location="cpu", weights_only=True))
            model = model.to(device).eval()
            if device.type == "cuda": model = model.to(memory_format=torch.channels_last)
            
            for j in range(len(stems)):
                p = predict(model, test_u8[j], stats[j, 0], stats[j, 1], device)
                with lock: total[j] += p

    threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
    for th in threads: th.start()
    for th in threads: th.join()

    rows = []
    u8_hi = int(round(float(KEEP["t_hi"]) * 255))
    for j, stem in enumerate(stems):
        avg_p = np.clip(np.round(total[j].astype(np.float32) / len(models_meta)), 0, 255).astype(np.uint8)
        lab, area, meanp, pmax = components(avg_p, KEEP["t_lo"])
        ok_labels = np.flatnonzero((pmax > u8_hi) & (area >= KEEP["min_area"]) & (meanp >= KEEP["min_meanp"])) + 1
        
        objs = ndi.find_objects(lab)
        for c in ok_labels:
            sl = objs[c - 1]
            m = np.zeros((H, W), np.uint8)
            m[sl] = lab[sl] == c
            rle = mask_util.encode(np.asfortranarray(m))["counts"].decode("ascii")
            rows.append((f"{stem}_{c}", rle))

    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    sub.to_csv(os.path.join(output_path, "submission.csv"), index=False)
    print("submission.csv generated successfully.")

if __name__ == "__main__":
    main()
