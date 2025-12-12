# LoRA-Depth Fall Detection

Privacy-preserving fall detection using LoRA-adapted depth estimation from RGB cameras.

## Project Overview

This project explores using **LoRA-adapted Depth Anything V2** for elderly fall detection. The key idea is that RGB cameras (cheap and ubiquitous) can estimate depth maps, which are then used for fall detection — preserving privacy by never storing or transmitting identifiable RGB frames.

### Architecture

```
RGB Frame → Depth Anything V2-Small (LoRA) → Depth Map
                                                  ↓
                                       Temporal Buffer (16 frames)
                                                  ↓
                                       Fall Detection Head (ConvLSTM)
                                                  ↓
                                       Fall / No-Fall + Confidence
```

### Key Contributions

1. **LoRA adaptation significantly improves depth estimation** for indoor scenes
2. **Efficient fine-tuning**: Only 2.32% trainable parameters
3. **Comprehensive ablation study** comparing RGB vs estimated depth modalities
4. **Dataset confound discovery** in UR-Fall dataset

---

## Results Summary

### Phase 1: Depth Estimation (NYU Depth V2)

LoRA adaptation dramatically improves indoor depth estimation:

| Configuration | Trainable % | abs_rel ↓ | δ1 ↑ | RMSE ↓ |
|---------------|-------------|-----------|------|--------|
| Zero-shot | 0% | 0.464 | 0.362 | 1.359 |
| r=8, QKV | 0.88% | 0.164 | 0.775 | 0.586 |
| r=8, QKV+dense | 1.18% | 0.152 | 0.803 | 0.563 |
| r=16, QKV | 1.75% | 0.161 | 0.804 | 0.566 |
| **r=16, QKV+dense** | **2.32%** | **0.142** | **0.824** | **0.541** |

**Key finding**: LoRA with r=16 targeting query, key, value, and dense layers achieves the best results with minimal trainable parameters.

### Phase 2: Fall Detection Ablation

#### UR-Fall Dataset (Deprecated)

All modalities achieved 100% accuracy due to a **dataset confound**:

```
ADL  mean depth: 0.4479 ± 0.0313
Fall mean depth: 0.5926 ± 0.0350
T-test p-value:  7.89e-67
```

The dataset is trivially separable by average depth value. Temporal shuffling had no effect on accuracy, confirming the model learns a single-frame shortcut rather than fall dynamics.

#### UP-Fall Dataset (Primary Evaluation)

| Condition | Accuracy | F1 | Precision | Recall |
|-----------|----------|-----|-----------|--------|
| RGB | **0.886** | **0.559** | 0.512 | 0.616 |
| LoRA Depth | 0.714 | 0.435 | 0.283 | 0.934 |

**Temporal Shuffle Test** (accuracy drop when frames shuffled):
- RGB: 0.0002 (no temporal dependency)
- Depth: -0.0018 (no temporal dependency)

### Critical Findings

1. **RGB outperforms estimated depth** on UP-Fall (contrary to hypothesis)
2. **Neither model learns temporal patterns** — shuffle test shows ~0% accuracy drop
3. **Depth model struggles with Activity 11 (Laying)** — predicts 98% as "Fall"
4. **ConvLSTM architecture may be insufficient** for learning fall dynamics

---

## Installation

```bash
# Clone repository
git clone https://github.com/pranavpatel08/Fall-detection-using-Depth-Anything-v2-LoRA.git
cd lora-depth-fall

# Create environment
conda create -n lora-depth python=3.10
conda activate lora-depth

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (tested on H200)
- ~80GB GPU memory for batch_size=512 training

---

## Dataset Setup

### NYU Depth V2 (for depth adaptation)

```bash
python data/scripts/download_nyu.py
```

Downloads to `data/nyu_depth_v2/nyu_depth_v2_labeled.mat` (1,449 RGB-D pairs).

### UP-Fall Dataset (for fall detection)

1. Download from [UP-Fall official source](http://sites.google.com/up.edu.mx/har-up/)
2. Extract to `data/up_fall/`
3. Generate estimated depth maps:

```bash
python scripts/generate_upfall_depth.py --resume
```

**Expected structure:**
```
data/up_fall/
├── Subject1/
│   ├── Activity1/
│   │   ├── Trial1/
│   │   │   ├── Camera1/
│   │   │   │   ├── *.png (RGB frames)
│   │   │   │   └── depth_lora/ (generated)
```

---

## Usage

### 1. Train Depth LoRA Adapter

```bash
python scripts/train_depth_lora.py
```

Configuration: `configs/depth_lora.yaml`

Best checkpoint saved to: `outputs/depth_lora/checkpoints/best/`

### 2. Generate Depth Maps for UP-Fall

```bash
# Full generation (H200 optimized)
python scripts/generate_upfall_depth.py --batch_size 48

# Resume if interrupted
python scripts/generate_upfall_depth.py --resume
```

### 3. Run Fall Detection Ablation

```bash
# Run both conditions in parallel (separate terminals)
python scripts/ablations/run_upfall_ablation.py --condition rgb
python scripts/ablations/run_upfall_ablation.py --condition depth_lora

# Merge results and run temporal shuffle test
python scripts/ablations/merge_ablation_results.py --test_shuffle
```

### 4. Diagnostics

```bash
# Analyze model predictions per activity
python scripts/ablations/run_ablation.py  # For UR-Fall diagnostics

# Detailed UP-Fall diagnosis
python scripts/visualizations/visualize_per_activity.py
```

---

## Project Structure

```
lora-depth-fall/
├── configs/
│   ├── depth_lora.yaml           # LoRA training config
│   └── fall_detection.yaml       # Fall detector config
│
├── data/
│   ├── scripts/                  # Data download/preparation
│   │   ├── download_nyu.py
│   │   ├── download_upfall.py
│   │   ├── download_urfall.py
│   │   ├── generate_estimated_depth.py
│   │   ├── generate_upfall_depth.py
│   │   ├── prepare_le2i.py
│   │   └── prepare_upfall_data.py
│   ├── nyu_depth_v2/
│   ├── up_fall/                  # Primary dataset
│   └── ur_fall/                  # Deprecated (confound)
│
├── outputs/
│   ├── ablation/
│   │   └── results.yaml          # UR-Fall results (deprecated)
│   ├── depth_lora/
│   │   ├── checkpoints/best/     # Best LoRA weights
│   │   └── config.yaml
│   ├── fall_detection/
│   │   ├── checkpoints/
│   │   ├── config.yaml
│   │   └── results.yaml
│   └── visualizations/
│
├── scripts/
│   ├── ablations/
│   │   ├── merge_ablation_results.py
│   │   ├── run_ablation.py
│   │   └── run_upfall_ablation.py
│   ├── visualizations/
│   │   ├── export_wandb_curves.py
│   │   ├── visualize_confusion_matrices.py
│   │   ├── visualize_nyu_comparison.py
│   │   ├── visualize_per_activity.py
│   │   ├── visualize_samples.py
│   │   └── visualize_upfall_samples.py
│   ├── poc_lora_depth.py
│   ├── train_depth_lora.py
│   └── train_fall_detector.py
│
├── src/
│   ├── data/
│   │   ├── nyu_dataset.py
│   │   ├── upfall_dataset.py
│   │   └── urfall_dataset.py
│   └── models/
│       ├── depth_lora.py
│       └── fall_detector.py
│
├── requirements.txt
└── README.md
```

---

## Lessons Learned

### What Worked

1. **LoRA adaptation is highly effective** — 2.3% parameters achieves 0.82 δ1 accuracy
2. **Systematic ablation revealed dataset issues** before false conclusions
3. **UP-Fall's Activity 11 (Laying)** provides a genuine test of temporal modeling

### What Didn't Work

1. **Depth estimation ≠ better fall detection** — RGB appearance cues may be more discriminative than depth geometry for this task
2. **ConvLSTM doesn't learn temporal patterns** on either modality — both models are effectively single-frame classifiers
3. **UR-Fall has a critical confound** — results on this dataset should not be trusted

### Future Directions

1. **Explore attention-based temporal models** (Video Transformers, TimeSformer)
2. **Frame-level annotation** for better supervision during fall events
3. **Motion features** (optical flow, depth flow) rather than raw depth
4. **Multi-camera fusion** for better spatial coverage
5. **Investigate why RGB outperforms depth** — may indicate depth maps lack fine-grained information

---

## Acknowledgments

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for the base depth model
- [UP-Fall Dataset](http://sites.google.com/up.edu.mx/har-up/) for fall detection evaluation
- [PEFT](https://github.com/huggingface/peft) for LoRA implementation