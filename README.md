# Substation Safety PPE Detection (Helmet, Gloves, Footwear) using SSD-VGG16 & DeMa Algorithm

This project implements the methodology from the IEEE 2023 research paper:
> **"Substation Safety Awareness Intelligent Model: Fast Personal Protective Equipment Detection Using GNN Approach"**  
> *Meng Zhao, Student Member, IEEE, and Masoud Barati, Senior Member, IEEE* (IEEE Transactions on Industry Applications, Vol. 59, No. 3, May/June 2023).

---

## 📌 Project Overview

This repository adapts the **Single Shot MultiBox Detector (SSD) with VGG-16 backbone**, **Few-Shot Data Augmentation**, and the **PPE DeMa (Detection & Matching) Algorithm** to detect and enforce safety compliance specifically for:
1. 🪖 **Helmet** (Hardhat)
2. 🧤 **Gloves** (Protective / Safety Gloves)
3. 🥾 **Footwear** (Leather Boots / Safety Shoes)

---

## ⚙️ Key Technical Features

### 1. Data Augmentation Few-Shot Strategy
To achieve robust performance with small dataset sizes (~50 images per class), the dataset pipeline applies geometric transformations, color space jittering, and multi-object context embedding.

### 2. MultiBox Loss Function (SSD with VGG-16 Backbone)
The objective loss function combines classification confidence loss $L_{\text{conf}}$ (Softmax) and localization loss $L_{\text{loc}}$ (Smooth L1):

$$L(x, c, l, g) = \frac{1}{N} \left( L_{\text{conf}}(x, c) + \alpha L_{\text{loc}}(x, l, g) \right)$$

where $N$ is the number of matched default boxes, and $\alpha = 1.0$.

### 3. PPE DeMa (Detection & Matching) Algorithm
- **Step 1 (PPE Determination)**: Runs object detection on input frame/image, producing bounding boxes, class labels, and similarity probability percentages ($79\% - 100\%$).
- **Step 2 (PPE Matching)**: Compares detected items against workplace safety requirements for **Helmet**, **Gloves**, and **Footwear**.
- **Step 3 (System Warning)**: If mandatory PPE items are missing, a warning alert is immediately triggered on screen and logged.

---

## 📁 Repository Structure

```
Substation-PPE-DeMa-Detection/
├── dema_engine/
│   ├── ppe_rules.py        # Safety standards & mandatory PPE sets (Helmet, Gloves, Footwear)
│   └── dema_algorithm.py   # 3-Step DeMa (Determination, Matching, Warning) Engine
├── models/
│   ├── ssd_vgg16.py        # PyTorch SSD + VGG-16 backbone model architecture & MultiBox loss
│   └── yolo_adapter.py     # Adapter for live YOLOv8 object detection integration
├── data/
│   ├── augmentation.py     # Few-shot data augmentation engine
│   └── dataset_loader.py   # Dataset loader (Pascal VOC XML / YOLO formats)
├── train.py                # Training loop with loss curve visualization
├── detect.py               # Real-time image/video/webcam detection script with alert banners
├── run_demo.py             # Verification script showcasing complete end-to-end pipeline
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Demo Verification
```bash
python run_demo.py
```
This generates sample test images, passes them through the SSD + DeMa pipeline, checks compliance for **Helmet**, **Gloves**, and **Footwear**, and saves visualization results to `output/`.

### 3. Run Inference on Custom Image / Video / Webcam
```bash
# Static Image
python detect.py --source sample.jpg

# Live Webcam Stream
python detect.py --source 0

# Video file
python detect.py --source input_video.mp4
```

### 4. Train Model
```bash
python train.py --epochs 20 --batch-size 4
```
Training automatically holds out `--val-split` (default 15%) of the images for
validation, applies a cosine-annealing LR schedule by default (`--lr-scheduler
{none,cosine,step}`), and saves two checkpoints to `models/`:
- `last_ssd_vgg16.pt` — latest epoch, used by `--resume`
- `best_ssd_vgg16.pt` — the epoch with the lowest validation loss so far (used by `detect.py` / `run_demo.py`)

Data augmentation (`data/augmentation.py`) applies color jitter plus geometric
transforms (random horizontal flip, random crop/scale jitter), keeping
bounding boxes aligned with the transformed image.

### 5. Evaluate Model (mAP)
```bash
python eval.py --weights models/best_ssd_vgg16.pt --data-dir data/sample_dataset
```
Reports mAP@0.5 and mAP@0.5:0.95 (COCO-style, all-point interpolation) plus
per-class AP on the same held-out validation split used during training —
using the same metric definitions Ultralytics reports for YOLOv8
(`mAP50`/`mAP50-95`), so results can be directly compared against a YOLOv8m
benchmark run on the same dataset.

---

## 📜 Citation & References
- Zhao, M., & Barati, M. (2023). *Substation Safety Awareness Intelligent Model: Fast Personal Protective Equipment Detection Using GNN Approach.* IEEE Transactions on Industry Applications, 59(3), 3142-3150.
