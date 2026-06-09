# ⚡ GridGuard: Edge-AI Substation Thermography & Automated Predictive Maintenance Infrastructure

An end-to-end intelligent computer vision pipeline that transforms raw thermal infrared telemetry into structured, real-time utility maintenance dispatches. This system combines **YOLOv8 Object Localization** with a **Deterministic Computer Vision Severity Engine** to monitor high-voltage electrical grid components without relying on high-compute cloud infrastructure.


---
GridGuard is an **end-to-end intelligent computer vision pipeline** that combines:

1. **Stage 1: Equipment Detection** (YOLOv8m - 96.4% mAP50)
   - Detects 5 electrical equipment types in thermal images
   - Localizes circuit breakers, disconnectors, transformers, surge arresters, wave traps

2. **Stage 2: Severity Scoring Engine** (Deterministic CV)
   - Analyzes thermal gradients within equipment regions
   - Calculates severity index based on grayscale intensity
   - Generates urgency levels: NORMAL, LOW, MEDIUM, CRITICAL

3. **Stage 3: Maintenance Dispatch** (Automated Alerts)
   - Creates structured work tickets with diagnostics
   - Recommends maintenance actions based on severity
   - Estimates time-to-failure and dispatch urgency

---

##  Key Features

 **Production-Ready Accuracy**
- Precision: 94.2%
- Recall: 92.6%
- mAP50: **96.4%**

 **Real-Time Dashboard**
- Streamlit-based web interface
- Live thermal image analysis
- Automated alert generation
- Visual hotspot indicators

 **Automated Maintenance Dispatch**
- Structured work tickets
- Equipment-specific recommendations
- Priority-based alert routing
- Time-to-failure estimates

---

## Performance Metrics

### Model 1: Equipment Detection (YOLOv8n)

| Metric | Value | 
|--------|-------|-------|
| **Precision** | 94.2% |
| **Recall** | 92.6% | 
| **mAP50** | **96.4%** | 
| **mAP50-95** | 62.6% |
| **Inference (GPU)** | 
| **Inference (CPU)** |

### Equipment Class Performance

| Equipment Type | Accuracy |
|---|---|---|
| Wave Traps | 100% | 
| Circuit Breakers | 95% |
| Disconnectors | 96% |
| Surge Arresters | 95% |
| Power Transformers | 87% |

---

##  Dashboard Preview

The Streamlit dashboard provides:

- **Real-time thermal image analysis**
- **Equipment localization with bounding boxes**
- **Automated hotspot detection and highlighting**
- **Severity scoring and alert generation**
- **Structured maintenance dispatch tickets**

---

##  System Architecture

┌─────────────────────────────┐
│   Thermal Image Input       │
│   (640x480, grayscale)      │
└────────────┬────────────────┘
             │
    ┌────────▼────────┐
    │ YOLOv8 Equipment│  (Stage 1: Detection)
    │   Detector      │
    │  96.4% mAP50    │
    └────────┬────────┘
             │
    ┌────────▼──────────────┐
    │  Severity Scoring     │  (Stage 2: Analysis)
    │  Engine (CV-based)    │
    │  Deterministic Rules  │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────────┐
    │ Maintenance Dispatch      │  (Stage 3: Action)
    │ Generator & Alert System  │
    └───────────────────────────┘
---

## Training & Evaluation

The model was trained on thermal substation imagery:

**Training Configuration:**
- Model: YOLOv8n (nano) → Small, fast, edge-friendly
- Epochs: 30
- Batch Size: 16
- Input Resolution: 640x480
- Device: GPU (CUDA)

**Ablation Study:**
- Baseline (YOLOv8n): 81.5% mAP50
- YOLOv8s: +3-4% improvement
- Final YOLOv8n with tuning: 96.4% mAP50 

```
### Key Technologies

- **YOLOv8** - Object Detection
- **OpenCV** - Image Processing  
- **Streamlit** - Web Dashboard
- **NumPy** - Numerical Computing
- **Python 3.8+** - Programming Language