# Face and Object Detection — Hybrid Haar Cascade + YOLO11 System

A real-time computer vision system that combines lightweight **Haar Cascade face detection** with the deep-learning **YOLO11** object detector. The Haar Cascade acts as a fast "gatekeeper": it checks every frame for a face first, and only triggers the heavier YOLO11 model when no face is found. This hierarchical design keeps the pipeline responsive even on devices without powerful GPUs.


## Overview

Most vision systems treat face detection and object detection as two separate, independently-run tasks — wasting compute by running both on every frame regardless of what's actually in the scene. This project proposes a **sequential, conditional pipeline** instead:

1. A grayscale frame is passed through a Haar Cascade face detector (fast, low-resource).
2. If a face **is** found → draw a bounding box, label it `Face Detected`, and skip the heavy model entirely.
3. If **no face** is found → trigger YOLO11 to scan the frame for everyday objects (from the 80 COCO classes) and label them with bounding boxes and confidence scores.

This avoids running both models on every frame and keeps the system real-time on modest hardware.

## Motivation

- Traditional deep learning detectors are accurate but computationally expensive — not ideal for continuous, real-time monitoring.
- Classical detectors (like Haar Cascades) are fast and cheap but less accurate, especially in complex scenes.
- Combining the two — cheap filtering first, expensive inference only when needed — gives a good speed/accuracy trade-off for use cases like home monitoring, classroom demos, and entry-level automation.

## How It Works

```
            ┌─────────────┐
            │  Open App   │
            └──────┬──────┘
                   ▼
            ┌─────────────┐
            │    Input    │
            └──────┬──────┘
                   ▼
            ┌─────────────┐
            │ Pre-process │
            └──────┬──────┘
                   ▼
            ┌─────────────┐
            │ Frame Input │
            └──────┬──────┘
                   ▼
        ┌────────────────────┐
        │  Haar Face Detector│
        └──────────┬─────────┘
                   ▼
              Face Found?
             ┌────┴─────┐
            Yes          No
             │            ▼
             │     ┌────────────────┐
             │     │ Trigger YOLO11 │
             │     └───────┬────────┘
             ▼              ▼
      ┌──────────────┐ ┌────────────────┐
      │ Face Detected│ │ Object Detected│
      └──────────────┘ └────────────────┘
```

**Decision logic priorities:**
- Face detection always runs first and takes priority.
- YOLO11 is only invoked when no face is present, reducing computational load.
- Output never overlaps — the system either shows a face result or an object result per frame, never both.

## Architecture

| Stage | Algorithm | Depends on |
|---|---|---|
| Face Detection | Haar Cascade (OpenCV, pre-trained) | Integral images, cascaded classifiers, AdaBoost |
| Object Detection | YOLO11 (Ultralytics, pre-trained on COCO) | Single-pass grid-based bounding box + class prediction |

**Why YOLO11 over alternatives:**
- vs. two-stage detectors (Faster R-CNN, Mask R-CNN): YOLO11 predicts boxes and classes in one pass, suited to real-time use rather than batch accuracy optimization.
- vs. other single-stage detectors (SSD, RetinaNet): YOLO11's anchor-free design simplifies training and offers a stronger speed/accuracy balance.
- vs. transformer-based detectors (DETR): YOLO11 is far less demanding on memory/compute, making it practical for lightweight, real-time systems rather than cloud-scale deployments.
- YOLO11 also uses ~22% fewer parameters than YOLOv8 while achieving higher mAP on COCO.

## Datasets & Models

- **Face Detection:** Pre-trained Haar Cascade classifiers from OpenCV (trained on positive/negative face vs. background image sets).
- **Object Detection:** YOLO11 pre-trained on the **COCO** (Common Objects in Context) dataset — 330,000+ images across 80 everyday object classes (bottle, phone, sofa, chair, etc.).
- **Face detection evaluation set:** 28 manually-labeled images sampled from the WIDER Face dataset.

## Preprocessing

| Step | Haar Cascade (Face) | YOLO11 (Object) |
|---|---|---|
| Color space | Grayscale (1 channel) | RGB (3 channels) |
| Resizing | Standard downscaling | Letterbox resizing (preserves aspect ratio) |
| Normalization | None | Scaled to 0.0–1.0 |
| Input structure | 2D array (H, W) | 4D tensor (Batch, Channels, H, W) |

## Evaluation Metrics

**Object detection:**
- **FPS (Frames Per Second)** — measures real-time throughput; compared conceptually between Haar-only and YOLO-only modes.
- **IoU (Intersection over Union)** — localization accuracy; predictions with IoU > 0.5 count as true positives.
- **mAP (Mean Average Precision)** — average precision across all 80 COCO classes at IoU = 0.5.


**Face detection (binary classification):** Confusion Matrix, Precision, Recall, Accuracy, F1-Score.

## Results

### Face Detection (Haar Cascade)
Evaluated on 28 labeled images from WIDER Face:

| Metric | Score |
|---|---|
| Precision | 0.86 |
| Recall | 0.75 |
| F1-Score | 0.80 |
| Accuracy | 0.68 |

- True Positives: 18, False Negatives: 6, False Positives: 3, True Negatives: 1
- Strong at detecting clear, front-facing faces — including a face partially covered by a mask.
- Struggles in crowded/complex scenes with small, side-viewed, or partially occluded faces.
- **Known failure case:** holding a phone close to the face (cheek/jawline) often produces an inaccurate bounding box that merges face and phone regions.

### Object Detection (YOLO11)
- Correctly identified multiple objects in a single frame (e.g. couch, bottle, book, mouse) with tight, accurate bounding boxes and confidence scores.
- **Known failure case:** a book lying flat on a couch was misclassified as a "laptop" (confidence 0.82) — likely due to shared flat, rectangular visual features between the two classes.
- Reported benchmarks (from literature, not measured live in this project): YOLO11n achieves ~39.5 mAP at ~1.5ms latency on T4 GPU hardware; inference speeds up to 290 FPS reported in other applications such as autonomous driving.

## Limitations

- Haar Cascade has reduced accuracy compared to modern CNN-based face detectors (e.g. RetinaFace, MTCNN), particularly under pose variation, poor lighting, and occlusion.
- YOLO11 can confuse visually similar object classes (e.g. books vs. laptops) when viewed from certain angles.
- Metrics reported are based on a small evaluation set and estimated/observed runtime performance rather than full real-time benchmarking.

## Future Work

- **Hybrid face detection:** combine Haar Cascade with SSD or other CNN-based refinement to validate/improve initial detections while keeping speed reasonable.
- **Data augmentation:** expand training data with difficult lighting, occlusion, and pose variation; consider generative models to synthesize diverse face examples.
- **Adaptive feature normalization** to improve generalization to unseen conditions (e.g. partial occlusion by handheld objects).
