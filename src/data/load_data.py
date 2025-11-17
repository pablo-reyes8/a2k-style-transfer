import os, io, random, time
from collections import defaultdict
import numpy as np
from typing import Optional
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torchvision import transforms
from PIL import Image as PILImage
from datasets import load_dataset
from datasets import Image as HFImage
import math
from collections import defaultdict

# ======================================================
# Configuración
# ======================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True

BATCH_SIZE   = 45
PIN_MEMORY   = True
DROP_LAST    = True
SEED         = 7

# Tamaño de entrenamiento
SIZE       = 256
FINAL_SIZE = 252  # usamos RandomResizedCrop directo a 252

# Targets de reducción
CONTENT_KEEP = 10_000   # COCO
STYLE_KEEP   = 5_000    # objetivo aprox de estilos buenos

# Normalización ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Auto-select de workers (Kaggle suele tener 2 vCPU)
CPU_COUNT   = os.cpu_count() or 2
NUM_WORKERS = 2 if CPU_COUNT <= 2 else min(4, CPU_COUNT - 1)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# Semillas
# ======================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_seed_init(worker_id: int):
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)

set_seed(SEED)

# ======================================================
# Transforms (content / style)
# ======================================================
content_tf = transforms.Compose([
    transforms.RandomResizedCrop(FINAL_SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

style_tf = transforms.Compose([
    transforms.RandomResizedCrop(FINAL_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ======================================================
# Utilidades HF / PIL
# ======================================================
def detect_image_col(ds):
    for c in ["image", "coco_url", "url", "image_url", "filepath", "file_name", "path"]:
        if c in ds.column_names:
            return c
    raise RuntimeError(f"No image-like column found. Columns: {ds.column_names}")

def is_valid_image_entry(x):
    if x is None:
        return False
    if isinstance(x, list):
        if len(x) == 0:
            return False
        x = x[0]
    if hasattr(x, "size"):  # PIL.Image
        return True
    if isinstance(x, dict):
        return ("bytes" in x and x["bytes"] is not None) or ("path" in x and x["path"])
    if isinstance(x, (bytes, bytearray)):
        return True
    if isinstance(x, str):
        return x.startswith("http") or os.path.exists(x)
    return False

def filter_valid_images(ds, img_key):
    return ds.filter(lambda ex: is_valid_image_entry(ex[img_key]), num_proc=1)

def cast_to_image(ds, col_name):
    if col_name != "image":
        ds = ds.cast_column(col_name, HFImage(decode=True))
        ds = ds.rename_column(col_name, "image")
    else:
        ds = ds.cast_column("image", HFImage(decode=True))
    return ds

def to_pil(x):
    if isinstance(x, list) and len(x) > 0:
        x = x[0]
    if hasattr(x, "size"):  # PIL
        return x.convert("RGB")
    if isinstance(x, dict) and "bytes" in x and x["bytes"] is not None:
        return PILImage.open(io.BytesIO(x["bytes"])).convert("RGB")
    if isinstance(x, dict) and "path" in x and x["path"]:
        return PILImage.open(x["path"]).convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return PILImage.open(io.BytesIO(x)).convert("RGB")
    if isinstance(x, str) and (x.startswith("http") or os.path.exists(x)):
        return PILImage.open(x).convert("RGB")
    raise TypeError(f"No pude convertir a PIL: tipo={type(x)}")

# ======================================================
# Dataset wrapper + collate
# ======================================================
class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, img_key: str, transform):
        self.ds = hf_ds
        self.img_key = img_key
        self.tfm = transform

    def __len__(self):
        return self.ds.num_rows

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = to_pil(ex[self.img_key])
        return self.tfm(img)

def collate_pixels(batch):
    return torch.stack(batch, dim=0)

# ======================================================
# DataLoader optimizado
# ======================================================
def make_loader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                pin=PIN_MEMORY, drop_last=DROP_LAST, collate_fn=collate_pixels):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=drop_last,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=worker_seed_init,
    )

# ======================================================
# Iterador dual (content + style)
# ======================================================
def infinite_cycle(loader):
    while True:
        for batch in loader:
            yield batch

class DualBatchIterator:
    def __init__(self, content_loader, style_loader):
        self.content_loader = content_loader
        self.style_iter = infinite_cycle(style_loader)

    def __iter__(self):
        for xb_content in self.content_loader:
            xb_style = next(self.style_iter)
            if xb_style.size(0) != xb_content.size(0):
                b = xb_content.size(0)
                if xb_style.size(0) > b:
                    xb_style = xb_style[:b]
                else:
                    idx = torch.randint(0, xb_style.size(0), (b,), device=xb_style.device)
                    xb_style = xb_style.index_select(0, idx)
            yield xb_content, xb_style

def make_train_iterator(content_loader, style_loader):
    return DualBatchIterator(content_loader, style_loader)

# ======================================================
# Recorte (uniforme) de datasets
# ======================================================
def truncate_dataloaders(content_loader: DataLoader,
                         style_loader: Optional[DataLoader] = None,
                         n: int = 60000,
                         seed: int = 77):
    rng = np.random.default_rng(seed)

    def sample_indices(ds_len, n_keep):
        n_keep = min(int(n_keep), int(ds_len))
        return rng.choice(ds_len, size=n_keep, replace=False)

    def shrink_dataset_from_loader(loader, n_keep):
        ds = loader.dataset
        if hasattr(ds, "ds"):  # nuestro HFDataset envuelve a HF ds como ds.ds
            base = ds.ds
            idx = sample_indices(len(base), n_keep)
            base_small = base.select(idx.tolist())
            ds_small = type(ds)(base_small, img_key=ds.img_key, transform=ds.tfm)
        else:
            idx = sample_indices(len(ds), n_keep)
            ds_small = Subset(ds, idx)

        return make_loader(
            ds_small,
            batch_size=loader.batch_size,
            num_workers=loader.num_workers,
            pin=loader.pin_memory,
            drop_last=loader.drop_last,
            collate_fn=loader.collate_fn,
        )

    content_small = shrink_dataset_from_loader(content_loader, n)
    if style_loader is None:
        return content_small
    style_small = shrink_dataset_from_loader(style_loader, n)
    return content_small, style_small

# ======================================================
# Submuestreo estratificado
# ======================================================
def stratified_pick(hf_ds, group_col: str, target_total: int, seed: int = 77,
                    min_per_group: int = 1):
    if group_col not in hf_ds.column_names:
        raise ValueError(f"'{group_col}' no está en {hf_ds.column_names}")
    rng = np.random.default_rng(seed)

    buckets = defaultdict(list)
    for i, g in enumerate(hf_ds[group_col]):
        if g is not None:
            buckets[g].append(i)

    G = len(buckets)
    if G == 0:
        # fallback: muestreo uniforme
        n = min(target_total, len(hf_ds))
        return rng.choice(len(hf_ds), size=n, replace=False).tolist()

    # per-group ideal (redondeo hacia arriba para no quedarnos cortos)
    per_group = max(min_per_group, math.ceil(target_total / G))

    picked = []
    for _, idxs in buckets.items():
        if len(picked) >= target_total:
            break
        if len(idxs) <= per_group:
            picked.extend(idxs)
        else:
            picked.extend(rng.choice(idxs, size=per_group, replace=False).tolist())

    # si nos pasamos un poco, recortamos
    if len(picked) > target_total:
        picked = rng.choice(picked, size=target_total, replace=False).tolist()

    return sorted(picked)

def add_brightness_stats(example):
    img = to_pil(example[wiki_img_col]).convert("RGB").resize((64, 64))
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    example["bright_mean"] = float(gray.mean())
    example["bright_std"]  = float(gray.std())
    return example


def keep_reasonable_brightness(example):
    bm = example["bright_mean"]
    bs = example["bright_std"]
    # ni muy oscuras ni lavadas
    if not (0.15 < bm < 0.90):
        return False
    # con algo de contraste
    if bs < 0.05:
        return False
    return True




