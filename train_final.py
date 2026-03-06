"""
train_final.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Continues from the saved checkpoint and runs
a stronger Stage 3 + Stage 4 to push past 85%.

Run: python train_final.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, copy, warnings
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.amp import GradScaler, autocast

warnings.filterwarnings("ignore")   # suppress deprecation noise

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob    = torch.nn.functional.log_softmax(pred, dim=-1)
        loss        = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        return ((1 - self.smoothing) * loss + self.smoothing * smooth_loss).mean()


def mixup_data(x, y, alpha=0.3):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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
                scaler.step(optimizer)      # optimizer BEFORE scheduler
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()        # scheduler AFTER optimizer

        _, preds          = torch.max(outputs, 1)
        running_loss     += loss.item() * grad_accum_steps * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    return running_loss / dataset_size, running_corrects.double() / dataset_size


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":

    # ── Paths ──────────────────────────────────
    data_dir  = r"C:\PLACEMENT CHALLENGE\PROJECTS\COMPUTER VISION\archive (1)\ECOMMERCE_PRODUCT_IMAGES"
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")
    CKPT_PATH = "efficientnet_b2_ecommerce_best.pth"

    # ── Config ─────────────────────────────────
    BATCH_SIZE    = 16
    GRAD_ACCUM    = 2       # effective batch = 32
    NUM_WORKERS   = 4

    # New stages to run on top of the 79.9% checkpoint
    STAGE3_EPOCHS = 8       # stronger Stage 3  (was 5, LR was too small)
    STAGE4_EPOCHS = 5       # new Stage 4: full model, cosine restart

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # ── Transforms ────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ── Data ───────────────────────────────────
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    # ── Device ────────────────────────────────
    device = torch.device("cuda")
    print(f"Device  : {torch.cuda.get_device_name(0)}")
    print(f"Classes : {num_classes}  |  Starting from checkpoint: {CKPT_PATH}\n")

    # ── Load checkpoint ────────────────────────
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = models.efficientnet_b2()
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    criterion    = LabelSmoothingCrossEntropy(smoothing=0.1)
    scaler       = GradScaler("cuda")
    best_val_acc = torch.tensor(ckpt["best_val_acc"])
    best_weights = copy.deepcopy(model.state_dict())

    print(f"Loaded checkpoint  →  previous best val acc: {best_val_acc:.4f}")

    # ══════════════════════════════════════════
    # STAGE 3 (redo) — Full model, LR=5e-5
    # Key fix: previous run used 5e-6 (too small)
    # ══════════════════════════════════════════
    print(f"\n🔁 STAGE 3 (stronger): Full model fine-tune  lr=5e-5  ({STAGE3_EPOCHS} epochs)")
    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = OneCycleLR(optimizer, max_lr=5e-5,
                           steps_per_epoch=len(train_loader) // GRAD_ACCUM,
                           epochs=STAGE3_EPOCHS, pct_start=0.2,
                           anneal_strategy="cos")

    for epoch in range(STAGE3_EPOCHS):
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
        print(f"  Epoch {epoch+1}/{STAGE3_EPOCHS} | "
              f"Train {tr_loss:.4f}/{tr_acc:.4f} | Val {vl_loss:.4f}/{vl_acc:.4f} {tag}")

    # ══════════════════════════════════════════
    # STAGE 4 — Cosine restart, LR=1e-5
    # Helps escape any local minima
    # ══════════════════════════════════════════
    print(f"\n🎯 STAGE 4: Cosine restart fine-tune  lr=1e-5  ({STAGE4_EPOCHS} epochs)")

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-8)

    for epoch in range(STAGE4_EPOCHS):
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
        print(f"  Epoch {epoch+1}/{STAGE4_EPOCHS} | "
              f"Train {tr_loss:.4f}/{tr_acc:.4f} | Val {vl_loss:.4f}/{vl_acc:.4f} {tag}")

    # ── Save ───────────────────────────────────
    torch.save({
        "model_state_dict": best_weights,
        "class_names":      class_names,
        "num_classes":      num_classes,
        "best_val_acc":     best_val_acc.item(),
    }, "efficientnet_b2_ecommerce_best.pth")

    print(f"\n✅ Best model saved  →  Val Accuracy: {best_val_acc:.4f}")
    if best_val_acc >= 0.85:
        print("🚀 Production ready! Deploy with the inference script below.")
    else:
        print("📈 Good progress — run this script once more to squeeze out more accuracy.")

    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFERENCE (paste into your Flask/FastAPI app)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image

def load_model(path="efficientnet_b2_ecommerce_best.pth"):
    ckpt  = torch.load(path, map_location="cpu")
    model = models.efficientnet_b2()
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, ckpt["num_classes"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["class_names"]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

model, class_names = load_model()
img  = Image.open("product.jpg").convert("RGB")
inp  = transform(img).unsqueeze(0)
with torch.no_grad():
    pred = model(inp).argmax(1).item()
print("Category:", class_names[pred])
""")