import os
from typing import Dict, List, Tuple, Union
from PIL import Image
import random

import torch
from torchvision import transforms

from config import Config

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PILImage = Image.Image

def discover_images(root_dir: str) -> Dict[str, List[str]]:
    """Return {class_name: [image_paths]} discovered from a folder-of-folders structure."""
    class_to_paths: Dict[str, List[str]] = {}
    for cls in sorted(os.listdir(root_dir)):
        cdir = os.path.join(root_dir, cls)
        if not os.path.isdir(cdir):
            continue
        paths = []
        for fn in os.listdir(cdir):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(cdir, fn))
        if paths:
            class_to_paths[cls] = sorted(paths)
    if not class_to_paths:
        raise RuntimeError(f"No classes with images found in: {root_dir}")
    return class_to_paths

def preload_images(class_to_paths: Dict[str, List[str]]) -> Dict[str, List[PILImage]]:
    """Load all images once into RAM as RGB PILs to remove disk I/O in training."""
    cached: Dict[str, List[PILImage]] = {}
    total = 0
    for cls, paths in class_to_paths.items():
        imgs = []
        for p in paths:
            imgs.append(Image.open(p).convert("RGB"))
        cached[cls] = imgs
        total += len(imgs)
    print(f"[data] RAM-cached {total} images across {len(cached)} classes.")
    return cached

def default_transforms(cfg: Config, split: str):
    if split == "train":
        aug = [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)], p=0.7),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(cfg.normalize_mean, cfg.normalize_std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        ]
    else:
        aug = [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.normalize_mean, cfg.normalize_std),
        ]
    return transforms.Compose(aug)

def sample_episode(
    data_dict: Dict[str, List[Union[str, PILImage]]],
    n_way: int,
    k_shot: int,
    q_queries: int,
    split: str,
    cfg: Config,
):
    """Return support and query tensors + labels for one episode. Works with paths OR preloaded PILs."""
    assert n_way <= len(data_dict), "n_way exceeds number of classes."
    classes = random.sample(list(data_dict.keys()), n_way)

    supp_imgs, supp_labels, qry_imgs, qry_labels = [], [], [], []
    tform = default_transforms(cfg, "train" if split == "train" else "eval")

    for ci, cls in enumerate(classes):
        items = data_dict[cls]
        assert len(items) >= k_shot + q_queries, f"Not enough images in class {cls} for k_shot+q_queries"
        picks = random.sample(items, k_shot + q_queries)
        supp, qry = picks[:k_shot], picks[k_shot:]

        for obj in supp:
            img = Image.open(obj).convert("RGB") if isinstance(obj, str) else obj
            supp_imgs.append(tform(img))
            supp_labels.append(ci)
        for obj in qry:
            img = Image.open(obj).convert("RGB") if isinstance(obj, str) else obj
            qry_imgs.append(tform(img))
            qry_labels.append(ci)

    supp_x = torch.stack(supp_imgs, dim=0)  # [n_way*k_shot, C, H, W]
    qry_x  = torch.stack(qry_imgs,  dim=0)  # [n_way*q_queries, C, H, W]
    supp_y = torch.tensor(supp_labels, dtype=torch.long)  # [n_way*k_shot]
    qry_y  = torch.tensor(qry_labels,  dtype=torch.long)  # [n_way*q_queries]

    return (supp_x, supp_y, qry_x, qry_y, classes)
