# BraTS Brain Tumor Segmentation Project
## 10-Minute Presentation Speech

---

## PART 1: PROJECT NEED & MOTIVATION (2-3 minutes)

### Opening Statement

Good morning/afternoon everyone. Today, I'll be presenting our project: **"Hybrid CNN-Transformer Architecture for Brain Tumor Segmentation"** — an AI-powered medical imaging solution designed for the BraTS 2021 Challenge.

### The Problem: Why This Matters

Brain tumors, particularly **gliomas**, represent one of the most aggressive forms of cancer affecting the central nervous system. Let me share some critical facts:

- **Every year**, thousands of patients are diagnosed with brain tumors worldwide
- **Manual tumor segmentation** by radiologists takes **30 to 60 minutes per patient**
- There's significant **inter-observer variability** — different doctors may draw tumor boundaries differently
- **Time is critical** — faster diagnosis means faster treatment initiation

### Clinical Importance

Accurate brain tumor segmentation is essential for:

1. **Surgical Planning**: Neurosurgeons need precise tumor boundaries to maximize tumor removal while preserving healthy brain tissue

2. **Radiotherapy Planning**: Radiation oncologists require accurate segmentation to deliver targeted doses to tumor tissue while minimizing damage to healthy brain regions

3. **Treatment Response Assessment**: Tracking tumor volume changes over time helps doctors evaluate if treatment is working

4. **Prognosis Prediction**: Tumor volume and infiltration patterns directly correlate with patient survival outcomes

### Our Solution

We developed an **automated deep learning system** that:
- Reduces segmentation time from **30-60 minutes to just seconds**
- Provides **consistent, reproducible results** — eliminating inter-observer variability
- Achieves **89.1% accuracy** on Whole Tumor segmentation
- Includes a **web-based interface** for easy clinical use with **3D visualization**

---

## PART 2: MODEL ARCHITECTURE (4-5 minutes)

### Overview: Hybrid CNN-Transformer Design

Our model, called **OptimizedUNet3D**, is a **hybrid architecture** that combines the best of two worlds:

1. **Convolutional Neural Networks (CNNs)** — excellent at extracting local features like edges and textures
2. **Transformers** — powerful at capturing global context and long-range dependencies

Think of it this way: CNNs are like examining each tree in detail, while Transformers help us see the entire forest.

### Input Data: Multi-Modal MRI

Our model processes **4 MRI modalities** simultaneously:

| Modality | Purpose |
|----------|---------|
| **T1** | Shows anatomical structures |
| **T1ce** (Contrast-Enhanced) | Highlights active tumor regions |
| **T2** | Reveals edema and tumor boundaries |
| **FLAIR** | Shows the full extent of infiltration |

The input volume is **160 × 192 × 160 voxels** — essentially a 3D cube of brain data with 4 channels.

### Architecture Components

#### 1. Encoder Path (Feature Extraction)

The encoder progressively extracts hierarchical features:

```
Input (4 channels) → 48 → 96 → 192 → 384 → 768 channels
```

Each encoder block contains:
- **Two 3D Convolutions** (3×3×3 kernels) — extract local spatial features
- **Instance Normalization** — handles varying MRI intensities
- **GELU Activation** — smooth non-linear activation
- **Lightweight Attention** — refines important features
- **Max Pooling** (2×2×2) — reduces spatial dimensions by half

As we go deeper, we capture increasingly abstract features — from edges at the first level to semantic tumor characteristics at deeper levels.

#### 2. Transformer Bottleneck (Global Context)

This is the **key innovation** that distinguishes our architecture.

At the bottleneck (10×12×10 spatial resolution):
- Each spatial location becomes a **token**
- We have **1,200 tokens** total
- Each token has **768 features**

The Transformer consists of:
- **4 layers** of self-attention
- **8 attention heads** per layer
- Each position can attend to **all other positions**

**Why is this important?**
- Enhancing Tumor (ET) is typically surrounded by Edema
- Necrotic Core (NCR) appears within the enhancing ring
- These spatial relationships **cannot be captured by local convolutions alone**

#### 3. Decoder Path (Spatial Reconstruction)

The decoder reconstructs the spatial resolution:

```
768 → 384 → 192 → 96 → 48 → 4 output classes
```

Key features:
- **Transposed Convolutions** for learnable upsampling
- **Attention Gates** on skip connections — filter relevant features
- **Deep Supervision** — auxiliary outputs at each level for better gradient flow

#### 4. Attention Gates

Skip connections normally pass ALL encoder features. Our **Attention Gates** learn to:
- Suppress irrelevant normal brain tissue features
- Highlight tumor-relevant features
- Dramatically improve boundary accuracy

### Output: Tumor Sub-regions

The model segments **4 classes**:

| Class | Label | Clinical Meaning |
|-------|-------|------------------|
| Background | 0 | Healthy brain tissue |
| NCR (Necrotic Core) | 1 | Dead tissue in tumor center |
| ED (Edema) | 2 | Swelling around tumor |
| ET (Enhancing Tumor) | 4 | Active, aggressive region |

For the BraTS challenge, we compute:
- **Whole Tumor (WT)** = NCR + ED + ET — Total tumor burden
- **Tumor Core (TC)** = NCR + ET — Solid tumor mass
- **Enhancing Tumor (ET)** = ET only — Most aggressive region

### Loss Function: 9 Components

Our comprehensive loss function addresses multiple challenges:

| Component | Weight | Purpose |
|-----------|--------|---------|
| Dice Loss | 0.25 | Primary overlap metric |
| BraTS Region Loss | 0.20 | Directly optimizes WT/TC/ET |
| Boundary Loss | 0.15 | Improves edge accuracy (HD95) |
| Tversky Loss | 0.10 | Handles class imbalance |
| Lovász-Softmax | 0.10 | IoU optimization |
| Focal CE | 0.05 | Prevents over-prediction |
| Others | 0.15 | Anatomical constraints |

### Training Details

- **Hardware**: 4× NVIDIA A100 80GB GPUs on RunPod Cloud
- **Parameters**: 81.12 Million
- **Batch Size**: 2 per GPU with gradient accumulation (effective batch = 64)
- **Training Time**: ~300 epochs with early stopping
- **Mixed Precision**: FP16 for memory efficiency

### Results Achieved

| Metric | Whole Tumor | Tumor Core | Enhancing Tumor |
|--------|-------------|------------|-----------------|
| **Dice Score** | 90.77% | 78.86% | 66.41% |
| **HD95** | 8.24 mm | 38.40 mm | 85.75 mm |

Our **90.77% Whole Tumor Dice score** demonstrates that the model reliably identifies the complete tumor extent — crucial for treatment planning.

---

## PART 3: WEBSITE FLOW & USER INTERFACE (2-3 minutes)

### Overview: End-to-End Web Application

We built a complete **Next.js web application** called **"BraTS Brain Tumor Analysis"** that makes this technology accessible to clinicians.

### User Journey

#### Step 1: Authentication (Login/Signup Page)

- Users first land on the **Home Page** with a clean, professional interface
- New users can **Create an Account** with their credentials
- Existing users can **Sign In** securely
- All data is protected with **JWT authentication**

#### Step 2: Upload Page

Once authenticated, users access the **Upload Interface**:

1. **Enter Patient Details**:
   - Patient Name, Age, Gender
   - Weight, Height
   - Known disorders
   - Clinical description

2. **Upload MRI Scans**:
   - Drag-and-drop interface for **NIfTI files** (.nii or .nii.gz)
   - Only accepts valid MRI scan formats
   - Upload all 4 modalities (T1, T1ce, T2, FLAIR)

3. **Click "Process"**:
   - Real-time progress bar shows:
     - Creating session...
     - Uploading MRI scans...
     - Initializing AI model...
     - Running tumor segmentation...
     - Generating 3D model...
     - Creating medical report...

#### Step 3: 3D Viewer Page

After processing, users see the **Interactive 3D Visualization**:

**Left Panel — 3D Brain Viewer**:
- **Three.js-powered** 3D rendering
- View the brain with tumor regions highlighted in different colors:
  - **Gray**: Brain tissue (adjustable transparency)
  - **Dark Red**: Necrotic Core (NCR)
  - **Yellow**: Edema (ED)
  - **Bright Red**: Enhancing Tumor (ET)
- **Interactive Controls**:
  - Rotate, zoom, pan the 3D model
  - Toggle visibility of each region
  - Adjust brain opacity slider
  - Scale the model

**Right Panel — Medical Report**:
- **Tumor Analysis Summary**:
  - Whole Tumor Volume (cm³)
  - Tumor Core Volume
  - Enhancing Tumor Volume
  - Estimated tumor location (hemisphere, lobe)
  - Estimated tumor grade with confidence score
- **Clinical Findings**: AI-generated observations
- **Recommendations**: Suggested next steps
- **Disclaimer**: Reminding users this is a decision support tool

#### Step 4: Dashboard (Analysis History)

- Users can view **all previous analyses**
- Each session shows:
  - Patient name and date
  - Processing status (completed/processing/failed)
- Click any session to **revisit the 3D visualization and report**
- Provides a complete **audit trail** for clinical records

### Technical Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 14, React, TypeScript |
| **UI Components** | Material-UI (MUI) |
| **3D Visualization** | Three.js with STL/GLB loaders |
| **Backend API** | FastAPI (Python) |
| **AI Processing** | PyTorch on RunPod serverless |
| **Database** | MongoDB for user data & sessions |
| **Authentication** | JWT tokens |
| **Deployment** | Vercel (frontend) + RunPod (inference) |

---

## CONCLUSION (30 seconds)

### Summary

To summarize, we have developed:

1. **A state-of-the-art AI model** — Hybrid CNN-Transformer achieving 90.77% accuracy on brain tumor segmentation

2. **Clinical-grade visualization** — Interactive 3D viewer for surgeons and radiologists

3. **End-to-end web application** — From MRI upload to medical report in minutes, not hours

### Clinical Impact

- Reduces segmentation time from **30-60 minutes to seconds**
- Eliminates **inter-observer variability**
- Supports **surgical planning and radiotherapy targeting**
- Provides **consistent, reproducible results**

### Important Note

While our model achieves strong performance, it is designed as a **clinical decision support tool** — not for autonomous diagnosis. All results should be reviewed by qualified medical professionals.

Thank you for your attention. I'm happy to take any questions.

---

## APPENDIX: Quick Reference Cards

### Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Model Parameters | 81.12 Million |
| Input Size | 160 × 192 × 160 voxels |
| WT Dice Score | 90.77% |
| TC Dice Score | 78.86% |
| ET Dice Score | 66.41% |
| Transformer Layers | 4 |
| Attention Heads | 8 |
| Loss Components | 9 |

### Tumor Region Quick Reference

| Abbreviation | Full Name | What It Means |
|--------------|-----------|---------------|
| WT | Whole Tumor | Everything (NCR + ED + ET) |
| TC | Tumor Core | Solid mass (NCR + ET) |
| ET | Enhancing Tumor | Active/aggressive part |
| NCR | Necrotic Core | Dead tissue center |
| ED | Edema | Surrounding swelling |

### Architecture Flow Diagram (For Slides)

```
INPUT (4×160×192×160)
        ↓
┌───────────────────┐
│   ENCODER PATH    │  ← CNN: Local Features
│  48→96→192→384→768│
└───────────────────┘
        ↓
┌───────────────────┐
│   TRANSFORMER     │  ← Global Context
│   BOTTLENECK      │     8 heads × 4 layers
│   (10×12×10×768)  │
└───────────────────┘
        ↓
┌───────────────────┐
│   DECODER PATH    │  ← CNN: Reconstruction
│  768→384→192→96→48│     + Attention Gates
└───────────────────┘
        ↓
OUTPUT (4 classes × 160×192×160)
```

---

## TIMING GUIDE

| Section | Duration | Cumulative |
|---------|----------|------------|
| Opening & Problem | 1:00 | 1:00 |
| Clinical Importance | 1:00 | 2:00 |
| Our Solution | 0:30 | 2:30 |
| Architecture Overview | 0:30 | 3:00 |
| Input Data | 0:30 | 3:30 |
| Encoder Path | 1:00 | 4:30 |
| Transformer Bottleneck | 1:00 | 5:30 |
| Decoder & Attention Gates | 0:45 | 6:15 |
| Output & Loss Function | 0:45 | 7:00 |
| Training & Results | 0:30 | 7:30 |
| Website - Authentication | 0:20 | 7:50 |
| Website - Upload | 0:40 | 8:30 |
| Website - 3D Viewer | 0:50 | 9:20 |
| Website - Dashboard | 0:20 | 9:40 |
| Conclusion | 0:20 | 10:00 |

---

*Document prepared for BraTS Brain Tumor Segmentation Project Presentation*
