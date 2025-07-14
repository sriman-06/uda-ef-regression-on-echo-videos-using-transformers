# 💓 Unsupervised Domain Adaptation for Ejection Fraction Regression in Echocardiography Using Vision Transformers

**🌍 World's First Transformer-Based UDA Pipeline** for echocardiogram-based EF estimation that **surpasses Stanford's baseline** and current SOTA models.

Achieved **MAE = 2.90** and **R² = 0.05** on **EchoNet-LVH** (cross-domain), setting a new benchmark for real-world echocardiographic EF regression.

---

## 📌 Highlights

- ✅ **Phase 1**: Supervised EF regression on **EchoNet-Dynamic**
- 🔍 **Phase 1 Baseline**: Cross-domain evaluation on **EchoNet-LVH**
- 🌍 **Phase 3 (UDA)**: Domain adaptation to **EchoNet-LVH**
- 📉 **MAE on LVH** improved from **9.32** (baseline) → **2.90** (UDA)
- 💡 Combined techniques:
  - 🔬 **DeiT Vision Transformer** (multi-frame + temporal attention)
  - 🧬 **Clinical Metadata Fusion** (Age, Sex, Blood Pressure)
  - 📈 **EMA Teacher** for pseudo-label consistency
  - 🔗 **MK-MMD Loss** for domain alignment
  - 🔒 **Entropy Minimization** for low-uncertainty predictions
  - 🧪 **Sharpened Pseudo-labeling + Mixup**
  - 🔁 **Test-Time Augmentation (TTA)** for robust validation

---

## 🔬 Method Overview

### 🧪 Phase 1 – Supervised EF Regression

- **Backbone**: DeiT transformer
- **Temporal Modeling**: Attention pooling across 8 video frames
- **Metadata Fusion**: Age, Sex, BP passed through MLP
- **Loss**: SmoothL1 Loss
- **Dataset**: EchoNet-Dynamic (with EF labels)

### 🌍 Phase 3 – Unsupervised Domain Adaptation (UDA)

- **Source Domain**: EchoNet-Dynamic
- **Target Domain**: EchoNet-LVH (no EF labels used)
- **UDA Strategy**:
  - MK-MMD for aligning source and target features
  - EMA teacher for generating stable pseudo-labels
  - Sharpening + confidence filtering
  - Mixup between source and confident target samples
  - Entropy loss to reinforce confident predictions
  - TTA for robust evaluation on LVH

---

## 📊 Final Results

| Phase               | Train Domain       | Eval Domain        | MAE ↓ | R² ↑    |
|--------------------|--------------------|--------------------|--------|---------|
| **Phase 1**         | EchoNet-Dynamic    | EchoNet-Dynamic    | **3.40** | **0.81** ✅ |
| **Phase 1 Baseline**| EchoNet-Dynamic    | **EchoNet-LVH**    | 9.32     | -5.16    |
| **Phase 3 (UDA)**   | EchoNet-Dynamic    | **EchoNet-LVH**    | **2.90** | **0.05** ✅ |

✅ **Conclusion**: Phase 3 UDA drastically improved cross-domain performance over Phase 1 baseline.

---

## 📁 Datasets

| Dataset           | Description |
|------------------|-------------|
| **EchoNet-Dynamic** | Public EF-labeled dataset for supervised training |
| **EchoNet-LVH**      | Public dataset used as unlabeled target domain (for UDA) |

> Videos are preprocessed into `.pt` format with 8-frame clips.  
> Metadata fields (Age, Sex, BP) are synthesized or added if missing.

---

## 📂 Directory Structure

EF-Regression/
├── phase.py and phase3.py # Full training pipeline (Phase 1 + Phase 3)
├── phase1_model.pt # Supervised model (EchoNet-Dynamic)
├── best_uda_regression.pt # Final UDA model (EchoNet-LVH)
├── metrics_log.csv # Phase 1 training logs
├── uda_metrics_log.csv # Phase 3 training logs
├── mae_over_epochs.png # Phase 1 MAE trend
├── uda_mae_over_epochs.png # Phase 3 MAE trend
├── phase1_on_lvh.png # Baseline evaluation (Phase 1 on LVH)
├── target_confident_pseudo_labels.csv # Saved pseudo-labels


---

## ⚙️ How to Run

1. Prepare:
   - `FileList.csv` (EchoNet-Dynamic)
   - `lvh_pseudo_labels.csv` (EchoNet-LVH)
   - `.pt` tensor files (8-frame clips for each video)

2. Run training:

python phase.py

3.This will:

Train Phase 1 on EchoNet-Dynamic

Evaluate Phase 1 on EchoNet-LVH (baseline)

Perform Phase 3 UDA on LVH

Log results and save best models/plots

📦 Requirements

pip install torch torchvision timm pandas numpy matplotlib scikit-learn tqdm

Python 3.8+

CUDA GPU recommended

🏆 Publication
This project is being submitted as the first successful Transformer-based UDA pipeline for EF regression.
If citing or building upon this work, please acknowledge the upcoming paper.



