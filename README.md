# 🔍 RetailLens AI  
AI-Powered Retail Product Image Quality & Category Classification

RetailLens AI is a production-style computer vision pipeline designed to automate product image validation and categorization for e-commerce platforms.  
It combines deep learning classification with rule based image quality assessment to improve catalog consistency and user experience.

---

## 📌 Overview

RetailLens AI provides two core capabilities:

### 🏷️ Product Category Classification  
Classifies product images into **9 e commerce categories** using a fine-tuned EfficientNet-B2 model.

### 🔍 Image Quality Assessment  
Evaluates product images for:
- Resolution adequacy
- Blur detection
- Brightness correctness  

Returns:
- Quality Score (0–100)
- Pass / Fail decision

This project demonstrates real-world **model optimization, imbalance handling, and production deployment**.

---

## 🎯 Key Features

| Feature | Details |
|--------|--------|
| Model | EfficientNet-B2 (ImageNet pretrained) |
| Training Strategy | Progressive fine tuning + targeted retraining |
| Final Validation Accuracy | **85.96%** |
| Categories | 9 retail classes |
| Quality Checks | Resolution · Blur · Brightness |
| Deployment | Streamlit Cloud |
| Hardware | RTX 2050 (4.3 GB VRAM) |

---

## 🗂️ Categories

- Baby Products  
- Beauty & Health  
- Clothing, Accessories & Jewellery  
- Electronics  
- Grocery  
- Hobby, Arts & Stationery  
- Home & Kitchen Tools  
- Pet Supplies  
- Sports & Outdoor  

---

## 📊 Training Progression

| Stage | Strategy | Validation Accuracy |
|------|----------|--------------------|
| Baseline | ResNet18 | 75.30% |
| Stage 1 | EfficientNet head training | 60.16% |
| Stage 2 | Partial unfreeze | 79.49% |
| Stage 3 | Full fine-tune | 84.91% |
| Stage 4 | Cosine restart polish | 85.35% |
| Stage 5 | Targeted imbalance retraining | **85.96%** |

---

## 🎯 Targeted Retraining for Weak Classes

Initial analysis revealed lower performance in visually complex categories:

- HOME_KITCHEN_TOOLS — 70.3%
- SPORTS_OUTDOOR — 75.7%
- HOBBY_ARTS_STATIONERY — 76.7%
- BEAUTY_HEALTH — 78.2%

### Optimization Strategy

- Focal Loss (γ = 2.0)
- Class-weighted training
- Weighted random sampling
- Strong augmentation pipeline
- Cosine annealing fine-tuning stage

### Result

Overall accuracy improved:

**85.35% → 85.96%**

More importantly, per-class balance improved significantly.

This demonstrates **real model debugging and targeted performance engineering**.

---

## 🧠 Advanced Training Engineering

This project incorporates production-style deep learning optimization:

- Mixed precision training (AMP)
- Gradient accumulation for low-VRAM GPUs
- Progressive unfreezing
- Cosine warm restarts scheduling
- Focal loss for hard example learning
- Imbalance-aware sampling
- Per-class diagnostic evaluation
- Robust augmentation strategy

These techniques reflect **industry-grade model tuning practices**.

---

## 🔍 Image Quality Checker

Three weighted evaluation criteria:

| Check | Weight | Threshold |
|------|--------|----------|
| Sharpness | 40% | Laplacian variance > 80 |
| Resolution | 30% | ≥ 100 × 100 px |
| Brightness | 30% | Mean pixel value 40–230 |

### Dataset Observations

- Overexposed images common in Electronics
- Minor blur issues detected in multiple categories
- Lighting variation significant in lifestyle product shots

---

## ⚙️ System Architecture

Input Image  
→ EfficientNet Classification  
→ Quality Assessment Pipeline  
→ Prediction + Quality Score  
→ Streamlit UI

---
---

## 🛠️ Tech Stack

- PyTorch
- Torchvision
- OpenCV
- Pillow
- Streamlit
- NumPy

---

## 📈 Engineering Insights

Key performance improvements achieved by:

- Switching architecture (ResNet → EfficientNet)
- Progressive fine-tuning strategy
- Advanced augmentation
- Imbalance-aware retraining
- Scheduler optimization
- Mixed precision acceleration

---

## 🚀 Future Improvements

- Confusion matrix-guided retraining
- Class-specific augmentation policies
- EfficientNet-V2 experiments
- Knowledge distillation for edge deployment
- ONNX optimization
- Real-time inference API

---

## ⭐ Project Goal

To demonstrate **real-world computer vision engineering skills** including:

- Model optimization
- Data imbalance handling
- Performance diagnostics
- Production deployment

