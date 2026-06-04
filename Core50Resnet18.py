"""
core50_loader.py
================
Load CORe50 theo Class Incremental (NC) setting với Avalanche.
Backbone: frozen ResNet-18 pretrained ImageNet → 512-d features.

Cấu trúc:
  - CORE50RESNET18      : class chính, lazy-load từng task
  - init_backbone()     : khởi tạo frozen ResNet-18
  - extract()           : trích features theo batch lớn
  - demo()              : xuất kết quả minh hoạ
"""

import os
import random
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from avalanche.benchmarks.classic import CORe50

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
SEED        = 317
BATCH_SIZE  = 256
EXTRACT_BS  = 512   # batch size khi extract features (lớn để nhanh)
NUM_WORKERS = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CORE50_CLASSES = [
    "plug_adapter", "mobile_phone", "scissor",    "light_bulb",  "can",
    "glass",        "ball",         "marker",      "cup",         "remote_control",
    "paper",        "plate",        "keyboard",    "box",         "flat_iron",
    "radio",        "shoe",         "door_knob",   "glasses",     "hammer",
    "fork",         "headphone",    "paper_clip",  "tablet",      "bag",
    "flower_pot",   "ruler",        "pen",         "magnifier",   "teapot",
    "book",         "pot",          "hat",         "towel",       "hat2",
    "toy",          "calculator",   "iron",        "tree",        "pencil",
    "SD_card",      "fan",          "spoon",       "CD",          "lighter",
    "mouse",        "watch",        "toy2",        "plant",       "table_lamp",
]

# ═══════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


train_transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

eval_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


# ═══════════════════════════════════════════════════════════════
# BACKBONE
# ═══════════════════════════════════════════════════════════════
def init_backbone() -> nn.Module:
    """Frozen ResNet-18 pretrained ImageNet, output 512-d features."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()   # bỏ classifier head
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(DEVICE)


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def extract(model: nn.Module, dataset, batch_size: int = EXTRACT_BS):
    """
    Trích features toàn bộ dataset.
    Trả về:
        feats  : Tensor (N, 512) trên CPU
        labels : Tensor (N,)     trên CPU
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
    )
    feats, labels = [], []
    for data, target, *_ in loader:
        out = model(data.to(DEVICE))          # (B, 512)
        out = out.view(out.size(0), -1)
        feats.append(out.cpu())
        labels.append(target.cpu())
    return torch.cat(feats), torch.cat(labels)


# ═══════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════
class CORE50RESNET18():
    def __init__(self, experiences = 9):
        
        self.n_classes = 50
        self.n_features = 512
                
        self.train_features, self.test_features = create_features()


    def clone(dataset):
        """ Tạo bản sao dataset mà không khởi tạo lại object mới """
        
        # Sao chép DataLoader của train_features với generator giữ nguyên seed
        new_train_features = []
        for dl in dataset.train_features:
            gen_seed = dl.generator.initial_seed() if dl.generator is not None else None
            new_generator = torch.Generator()
            if gen_seed is not None:
                new_generator.manual_seed(gen_seed)

            new_dl = torch.utils.data.DataLoader(
                dl.dataset,
                batch_size=dl.batch_size,  # Giữ nguyên batch_size gốc
                shuffle=True,
                num_workers=dl.num_workers,
                generator=new_generator
            )
            new_train_features.append(new_dl)

        # Sao chép test_features (không cần shuffle, không cần generator)
        new_test_features = [
            torch.utils.data.DataLoader(
                dl.dataset,
                batch_size=dl.batch_size,
                shuffle=False,
                num_workers=dl.num_workers
            )
            for dl in dataset.test_features
        ]

        # Gán lại dataset để giữ nguyên object nhưng update thuộc tính
        dataset.train_features = new_train_features
        dataset.test_features = new_test_features

        return dataset  # Trả về object đã được cập nhật

def create_features(run: int = 0):
    """
    Trả về:
        train_features[i] : DataLoader features train của task i (chỉ classes mới)
        test_features[i]  : DataLoader features test của task i  (chỉ classes mới)
                            — KHÔNG tích lũy, giống SplitCIFAR10
    """
    set_seed(SEED)
    model = init_backbone()

    benchmark = CORe50(
        scenario        = "nc",
        run             = run,
        mini            = False,
        train_transform = train_transform,
        eval_transform  = eval_transform,
    )

    # Extract toàn bộ test set một lần duy nhất
    #print("Extracting full test features...")
    full_test_feats, full_test_labels = extract(
        model, benchmark.test_stream[0].dataset
    )
    full_test_feats  = full_test_feats.to(DEVICE)
    full_test_labels = full_test_labels.to(DEVICE)

    train_features = []
    test_features  = []

    for train_exp in benchmark.train_stream:
        t           = train_exp.current_experience
        new_classes = sorted(train_exp.classes_in_this_experience)

        #print(f"Task {t+1}/9 | classes: {new_classes} "
        #    f"| train samples: {len(train_exp.dataset)}")

        # ── Train features ──────────────────────────────────
        train_feats, train_lbls = extract(model, train_exp.dataset)
        train_set = torch.utils.data.TensorDataset(train_feats, train_lbls)
        train_features.append(torch.utils.data.DataLoader(
            train_set,
            batch_size  = BATCH_SIZE,
            shuffle     = True,
            num_workers = 0,
            generator   = torch.Generator().manual_seed(SEED),
        ))

        # ── Test features — chỉ classes của task i (không tích lũy) ──
        mask = torch.isin(
            full_test_labels,
            torch.tensor(new_classes, dtype=full_test_labels.dtype,
                        device=DEVICE),
        )
        test_set = torch.utils.data.TensorDataset(
            full_test_feats[mask].cpu(),
            full_test_labels[mask].cpu(),
        )
        test_features.append(torch.utils.data.DataLoader(
            test_set,
            batch_size  = BATCH_SIZE,
            shuffle     = False,
            num_workers = 0,
        ))

        #print(f"         | test  samples (task only): {mask.sum().item()}")

    del full_test_feats, full_test_labels
    torch.cuda.empty_cache()

    return train_features, test_features


if __name__ == '__main__':
    dataset = CORE50RESNET18()
    
