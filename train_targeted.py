"""
train_targeted.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Targeted retraining to fix weak classes:
  - HOME_KITCHEN_TOOLS    70.3%
  - SPORTS_OUTDOOR        75.7%
  - HOBBY_ARTS_STATIONERY 76.7%
  - BEAUTY_HEALTH         78.2%

Strategy:
  1. Class-weighted loss  → penalizes mistakes on weak classes more
  2. Extra augmentation   → harder training on weak class images
  3. Focal loss           → focuses on hard-to-classify examples
  4. Loads best checkpoint and continues training

Run: python train_targeted.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, copy, warnings
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.amp import GradScaler, autocast
from collections import Counter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# FOCAL LOSS — focuses on hard examples
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, pred, target):
        ce    = nn.functional.cross_entropy(pred, target,
                                            weight=self.weight,
                                            reduction="none")
        pt    = torch.exp(-ce)
        loss  = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ─────────────────────────────────────────────
# MIXUP
# ─────────────────────────────────────────────
def mixup_data(x, y, alpha=0.3):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def run_epoch(model, loader, dataset_size, criterion,
              optimizer=None, scheduler=None, scaler=None,
              device="cuda", use_mixup=False, is_train=True,
              grad_accum_steps=1):

    model.train() if is_train else model.eval()
    running_loss = running_corrects = 0

    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        with autocast("cuda"):
            if is_train and use_mixup:
                inputs, y_a, y_b, lam = mixup_data(inputs, labels)
                outputs = model(inputs)
                loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam) / grad_accum_steps
            else:
                outputs = model(inputs)
                loss    = criterion(outputs, labels) / grad_accum_steps

        if is_train:
            scaler.scale(loss).backward()
            if (i + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

        _, preds          = torch.max(outputs, 1)
        running_loss     += loss.item() * grad_accum_steps * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    return running_loss / dataset_size, running_corrects.double() / dataset_size


# ─────────────────────────────────────────────
# PER-CLASS ACCURACY
# ─────────────────────────────────────────────
def per_class_accuracy(model, loader, class_names, device):
    model.eval()
    correct = torch.zeros(len(class_names))
    total   = torch.zeros(len(class_names))
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, preds = torch.max(model(inputs), 1)
            for p, l in zip(preds, labels):
                correct[l] += (p == l).item()
                total[l]   += 1
    print("\n📊 Per-Class Accuracy:")
    print("─" * 45)
    for i, cls in enumerate(class_names):
        acc  = 100 * correct[i] / total[i]
        flag = " ⚠️" if acc < 80 else " ✅"
        print(f"  {cls:<35} {acc:5.1f}%{flag}")
    print("─" * 45)
    overall = 100 * correct.sum() / total.sum()
    print(f"  {'OVERALL':<35} {overall:5.1f}%\n")
    return overall.item()


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":

    # ── Paths ──────────────────────────────────
    data_dir  = r"C:\PLACEMENT CHALLENGE\PROJECTS\COMPUTER VISION\archive (1)\ECOMMERCE_PRODUCT_IMAGES"
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")
    CKPT_PATH = os.path.join(data_dir, "efficientnet_b2_ecommerce_best.pth")

    # ── Config ─────────────────────────────────
    BATCH_SIZE  = 16
    GRAD_ACCUM  = 2
    NUM_WORKERS = 0      # Windows safe
    EPOCHS_1    = 8      # weighted loss stage
    EPOCHS_2    = 5      # final polish

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # ── Stronger augmentation for weak classes ─
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # more aggressive crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=15),
        transforms.RandomGrayscale(p=0.08),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ── Datasets ───────────────────────────────
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transform)
    class_names   = train_dataset.classes
    num_classes   = len(class_names)

    print(f"Classes: {class_names}")

    # ── Class weights (inverse frequency) ─────
    # Weak classes get higher weight → model penalized more for mistakes
    class_counts  = Counter(train_dataset.targets)
    total_samples = sum(class_counts.values())

    # Base inverse-frequency weights
    base_weights  = [total_samples / class_counts[i] for i in range(num_classes)]

    # Extra boost for the 4 weak classes
    WEAK_CLASSES  = {
        "HOME_KITCHEN_TOOLS":    2.0,   # worst — 70.3%
        "SPORTS_OUTDOOR":        1.8,   # 75.7%
        "HOBBY_ARTS_STATIONERY": 1.6,   # 76.7%
        "BEAUTY_HEALTH":         1.4,   # 78.2%
    }
    for cls, boost in WEAK_CLASSES.items():
        if cls in class_names:
            idx = class_names.index(cls)
            base_weights[idx] *= boost
            print(f"  ↑ Boosted weight for {cls} (×{boost})")

    # Normalize weights
    w_sum        = sum(base_weights)
    class_weights = torch.tensor([w / w_sum * num_classes for w in base_weights], dtype=torch.float32)

    # ── Weighted sampler → oversample weak classes ──
    sample_weights = [class_weights[t].item() for t in train_dataset.targets]
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)

    # ── Device ────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ── Load checkpoint ────────────────────────
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model = models.efficientnet_b2()
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_f, num_classes)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Loaded : {CKPT_PATH}")
    print(f"Base val acc: {ckpt['best_val_acc']:.4f}\n")

    # ── Unfreeze full model ────────────────────
    for p in model.parameters():
        p.requires_grad = True

    criterion    = FocalLoss(gamma=2.0, weight=class_weights.to(device))
    scaler       = GradScaler("cuda")
    best_val_acc = ckpt["best_val_acc"]
    best_weights = copy.deepcopy(model.state_dict())


    # ══════════════════════════════════════════
    # STAGE 1 — Focal loss + weighted sampler
    # ══════════════════════════════════════════
    print(f"🎯 STAGE 1: Focal loss + class weighting ({EPOCHS_1} epochs)")
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-5)
    scheduler = OneCycleLR(optimizer, max_lr=3e-5,
                           steps_per_epoch=len(train_loader) // GRAD_ACCUM,
                           epochs=EPOCHS_1, pct_start=0.2)

    for epoch in range(EPOCHS_1):
        tr_loss, tr_acc = run_epoch(model, train_loader, len(train_dataset),
                                    criterion, optimizer, scheduler, scaler,
                                    device, use_mixup=True, is_train=True,
                                    grad_accum_steps=GRAD_ACCUM)
        vl_loss, vl_acc = run_epoch(model, val_loader, len(val_dataset),
                                    criterion, device=device, is_train=False)
        tag = "✅ best" if vl_acc > best_val_acc else ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
        print(f"  Epoch {epoch+1}/{EPOCHS_1} | "
              f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Val {vl_loss:.4f}/{vl_acc:.4f} {tag}")

    # ══════════════════════════════════════════
    # STAGE 2 — Cosine polish, no mixup
    # ══════════════════════════════════════════
    print(f"\n🌐 STAGE 2: Final polish ({EPOCHS_2} epochs)")
    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-8)

    for epoch in range(EPOCHS_2):
        tr_loss, tr_acc = run_epoch(model, train_loader, len(train_dataset),
                                    criterion, optimizer, scheduler, scaler,
                                    device, use_mixup=False, is_train=True,
                                    grad_accum_steps=1)
        vl_loss, vl_acc = run_epoch(model, val_loader, len(val_dataset),
                                    criterion, device=device, is_train=False)
        scheduler.step(epoch)
        tag = "✅ best" if vl_acc > best_val_acc else ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
        print(f"  Epoch {epoch+1}/{EPOCHS_2} | "
              f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Val {vl_loss:.4f}/{vl_acc:.4f} {tag}")

    # ── Save ───────────────────────────────────
    torch.save({
        "model_state_dict": best_weights,
        "class_names":      class_names,
        "num_classes":      num_classes,
        "best_val_acc":     best_val_acc if isinstance(best_val_acc, float) else best_val_acc.item(),
    }, os.path.join(data_dir, "efficientnet_b2_ecommerce_best.pth"))

    print(f"\n✅ Saved  →  Best Val Accuracy: {best_val_acc:.4f}")

    # ── Final per-class accuracy ───────────────
    model.load_state_dict(best_weights)
    per_class_accuracy(model, val_loader, class_names, device)