# 🔍 RetailLens AI
### Retail Product Image Quality & Category Classification for E-commerce

---

## 📌 Overview

**RetailLens AI** is an end-to-end computer vision pipeline for e-commerce product images. It combines two core capabilities:

- 🏷️ **Category Classification** — Classifies product images into 9 e-commerce categories using a fine-tuned EfficientNet-B2 model achieving **85.35% validation accuracy**
- 🔍 **Image Quality Assessment** — Automatically checks product images for resolution, sharpness, and brightness, returning a **Pass/Fail verdict with a quality score (0–100)**

Built as part of a placement challenge project to demonstrate real-world computer vision skills.

---

## 🎯 Features

| Feature | Details |
|---------|---------|
| Model Architecture | EfficientNet-B2 (pretrained on ImageNet) |
| Training Strategy | 3-stage progressive fine-tuning |
| Validation Accuracy | **85.35%** (up from 75.3% baseline ResNet18) |
| Categories | 9 product classes |
| Quality Checks | Resolution · Blur · Brightness |
| Quality Output | Pass/Fail + Score (0–100) |
| Deployment | Streamlit Cloud |

---

## 🗂️ Categories

| Class | Category |
|-------|----------|
| 👶 | Baby Products |
| 💄 | Beauty & Health |
| 👗 | Clothing, Accessories & Jewellery |
| 📱 | Electronics |
| 🛒 | Grocery |
| 🎨 | Hobby, Arts & Stationery |
| 🏠 | Home & Kitchen Tools |
| 🐾 | Pet Supplies |
| ⚽ | Sports & Outdoor |

---

## 📊 Results

### Classification
```
Stage 1 (head only)      →  60.16% val accuracy
Stage 2 (partial unfreeze) →  79.49% val accuracy
Stage 3 (full fine-tune)  →  84.91% val accuracy
Stage 4 (cosine restart)  →  85.35% val accuracy ✅
```

### Image Quality (val set — 3632 images)
```
CLASS                           IMGS   PASS    AVG SCORE
─────────────────────────────────────────────────────────
BABY_PRODUCTS                    282   78%      89/100
BEAUTY_HEALTH                    312   81%      91/100
CLOTHING_ACCESSORIES_JEWELLERY   278   79%      90/100
ELECTRONICS                      351   83%      91/100
GROCERY                         1033   93%      93/100
HOBBY_ARTS_STATIONERY            283   86%      91/100
HOME_KITCHEN_TOOLS               445   83%      90/100
PET_SUPPLIES                     327   83%      90/100
SPORTS_OUTDOOR                   321   88%      92/100
─────────────────────────────────────────────────────────
TOTAL                           3632   86%      91/100
```

---

## 🧠 Model Training

### Architecture
- **Base model**: EfficientNet-B2 pretrained on ImageNet
- **Classifier head**: `Dropout(0.4) → Linear(1408 → 9)`
- **Loss**: Label Smoothing Cross Entropy (smoothing=0.1)
- **Augmentation**: MixUp, RandomErasing, ColorJitter, RandomAffine
- **Mixed Precision**: AMP (autocast + GradScaler) for VRAM efficiency

### 3-Stage Progressive Fine-tuning
```
Stage 1 — Classifier head only     (lr=5e-3, 5 epochs,  OneCycleLR)
Stage 2 — Top 3 feature blocks     (lr=1e-4, 8 epochs,  CosineWarmRestarts)
Stage 3 — Full model fine-tune     (lr=5e-5, 8 epochs,  OneCycleLR)
Stage 4 — Cosine restart polish    (lr=1e-5, 5 epochs,  CosineWarmRestarts)
```

### Hardware
- GPU: NVIDIA GeForce RTX 2050 (4.3GB VRAM)
- Training time: ~90 minutes total

---

## 🔍 Image Quality Checker

Three checks with weighted scoring:

| Check | Weight | Threshold |
|-------|--------|-----------|
| Sharpness (Laplacian variance) | 40% | > 80.0 |
| Resolution | 30% | Min 100×100px |
| Brightness (mean pixel value) | 30% | 40 – 230 |

**Common issues found in dataset:**
- `Overexposed` — 54 images in ELECTRONICS alone (white background + bright lighting)
- `Blurry` — 5 images in ELECTRONICS


---

## 📁 Project Structure

```
RetailLens-AI/
├── app.py                              # Streamlit web app
├── train_final.py                      # Training script (EfficientNet-B2)
├── image_quality.py                    # Image quality checker
├── requirements.txt                    # Dependencies
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, Torchvision
- **Model**: EfficientNet-B2
- **Image Processing**: OpenCV, Pillow
- **Web App**: Streamlit
- **Deployment**: Streamlit Cloud + Google Drive (model hosting)

---

## 📈 Improvement Journey

```
Original ResNet18 (6 epochs):     75.30%
EfficientNet-B2 Run 1:            79.90%  (+4.6%)
After stronger fine-tuning:       84.91%  (+5.0%)
Final (cosine restart):           85.35%  (+0.4%)
```

Key improvements over baseline:
- Switched ResNet18 → EfficientNet-B2
- Added MixUp augmentation
- Label smoothing loss
- Mixed precision training (AMP)
- Progressive 4-stage fine-tuning
- Best checkpoint saving across all epochs

---





