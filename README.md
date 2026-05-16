# Part 2 — Computer Vision Problem Formulation and CNN Prototype

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow) ![Task](https://img.shields.io/badge/Task-Image%20Classification-purple) ![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

## 📌 Problem Statement

Automatically classify vehicle surface photographs into one of four damage categories:

| Class | Description |
|---|---|
| `normal` | No visible surface damage |
| `dent` | Impact depression on the body panel |
| `scratch` | Surface abrasion / scoring |
| `stain` | Paint discolouration / contamination |

**Business Application:** Automated quality inspection on a vehicle manufacturing line and insurance damage triage.

---

## 📂 Repository Structure

```
part-2-cnn-computer-vision/
│
├── README.md                              ← You are here
├── notebook.ipynb                         ← All 7 tasks
├── requirements.txt
├── labels.csv                             ← Dataset labels
│
├── images/
│   ├── normal/                            ← Provided real images
│   ├── dent/                              ← Generated at runtime
│   ├── scratch/                           ← Generated at runtime
│   └── stain/                             ← Generated at runtime
│
├── results/
│   ├── task2_sample_images.png            ← 4×4 image grid
│   ├── task2_class_distribution.png       ← Class balance chart
│   ├── task3_augmentation.png             ← Augmentation examples
│   ├── accuracy_loss_curves.png           ← ✅ Training history
│   ├── confusion_matrix.png               ← ✅ Confusion matrix
│   ├── per_class_accuracy.png             ← Per-class bars
│   └── feature_maps.png                   ← Conv1 feature maps
│
└── sample_predictions/
    └── prediction_outputs.png             ← ✅ 20 test predictions
```

---

## 🔍 Task 1 — Problem Type: Image Classification

**Why Image Classification?**

Each photograph shows a vehicle surface and is labelled as exactly one defect type. The goal is to assign the correct whole-image category label — this is the definition of **multi-class image classification**.

| Type | Use When | Our Case |
|---|---|---|
| **Image Classification** | One label per image | ✅ Correct |
| Object Detection | Bounding boxes needed | ❌ No box annotations |
| Semantic Segmentation | Pixel-level masks | ❌ No pixel labels |

---

## 📊 Task 2 — Dataset

| Property | Value |
|---|---|
| Total images | ~480 (120 per class) |
| Classes | 4 (normal, dent, scratch, stain) |
| Image dimensions | 64 × 64 × 3 (RGB) |
| Class balance | Balanced (~25% each) |
| Format | PNG |

---

## 🧠 Task 4 — CNN Architecture

```
Input  (64 × 64 × 3)
  │
  ├─ Block 1: Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
  ├─ Block 2: Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
  ├─ Block 3: Conv2D(128) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
  │
  ├─ Flatten
  ├─ Dense(256, ReLU) → Dropout(0.50)
  └─ Dense(4, Softmax)  ←  output probabilities
```

**Loss:** Categorical Cross-Entropy  
**Optimiser:** Adam (lr = 0.001)  
**Regularisation:** BatchNormalization + Dropout + EarlyStopping + ReduceLROnPlateau

---

## 🔬 Task 6 — CNN Concepts Explained

### What is Convolution?
A small **learnable filter** (e.g. 3×3 weights) slides across the image and computes a dot product at every position, producing a **feature map** that shows *where* a specific pattern (edge, gradient, texture) appears. Crucially, the filter weights are *learned from data* — the network automatically discovers which visual patterns are diagnostic for each damage class.

```
Input patch × Filter → scalar activation
[ 120 118 115 ]   [ -1  0  1 ]
[ 121 119 116 ] × [ -2  0  2 ] = edge detector
[ 119 118 114 ]   [ -1  0  1 ]
```

### Why is Pooling Used?
**MaxPooling** takes the maximum value in each `2×2` region, halving the spatial dimensions. This achieves three things:
1. **Compression** — fewer parameters → faster training and inference
2. **Translation invariance** — a dent slightly left or right of centre produces the same output
3. **Increased receptive field** — deeper layers effectively "see" larger image regions

### Why is ReLU Commonly Used?
`ReLU(x) = max(0, x)` is the dominant activation in CNNs because:
- **No vanishing gradient**: derivative is 1 for x > 0, so gradients flow well through many layers
- **Sparse activations**: negative values become zero → efficient, interpretable representations
- **Computationally trivial**: much faster than sigmoid or tanh
- **Works well in practice**: empirically outperforms other activations on most vision tasks

### Why CNNs Beat Feed-Forward Networks for Images?

| Property | Dense (FF) Network | CNN |
|---|---|---|
| Parameters | `64×64×3 × H` = millions | `32 × 3×3×3` = 864 for first layer |
| Spatial awareness | None — pixels are flattened | Filters detect local 2D patterns |
| Position sensitivity | Must relearn for every position | Same filter applied everywhere (weight sharing) |
| Hierarchy | No built-in structure | Layers build: edges → shapes → objects |
| Overfitting risk | Very high on images | Controlled by weight sharing + pooling |

---

## 💼 Task 7 — Business Use Case: Automotive Manufacturing QC

**Problem:** Manual visual inspection of vehicles takes ~8 minutes per unit and has a 4.5% defect escape rate.

**AI Solution:** Real-time CNN classifies ceiling-camera images in <200 ms, routing each vehicle to the correct repair station automatically.

| KPI | Before AI | After AI |
|---|---|---|
| Inspection time | 8 min | 0.2 sec |
| Defect escape rate | 4.5% | <0.8% |
| Inspector headcount | 12 | 3 auditors |
| Annual saving | — | ~₹65 Lakhs |

**Other applicable domains:** Insurance claims · Healthcare dermoscopy · Agriculture crop disease · Retail shelf monitoring

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | ~88–93% |
| Loss | Categorical Cross-Entropy |
| Evaluation | Confusion matrix + per-class accuracy + feature maps |

---

## 🚀 How to Run

### Google Colab
1. Upload `notebook.ipynb` + `labels.csv` + `images/normal/` folder
2. Runtime → Run all  
   *(Synthetic images for dent/scratch/stain are auto-generated in cell 2)*

### Local
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

---

## 📚 References
- LeCun et al. (1998). *Gradient-Based Learning Applied to Document Recognition.*
- Krizhevsky et al. (2012). *ImageNet Classification with Deep CNNs.* NeurIPS.
- TensorFlow Keras Docs: [tensorflow.org/api_docs](https://www.tensorflow.org/api_docs)
