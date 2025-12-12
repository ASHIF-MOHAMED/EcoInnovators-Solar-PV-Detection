# ☀️ Project Surya-Drishti: AI-Powered Solar Verification

![YOLOv11](https://img.shields.io/badge/Model-YOLOv11--Large-blue)
![Task](https://img.shields.io/badge/Task-Instance_Segmentation-green)
![Status](https://img.shields.io/badge/Status-Training-orange)

> **Governance-ready, auditable, and low-cost remote verification pipeline for the PM Surya Ghar: Muft Bijli Yojana.**

## 📖 Overview
This project addresses the challenge of verifying rooftop solar installations remotely. Instead of manual field inspections, we utilize **GeoAI** to analyze satellite/aerial imagery given a specific coordinate (Latitude, Longitude).

The system not only detects the presence of solar panels but **Quantifies** them (Area & Capacity) using precise Instance Segmentation.

---

## 🎯 Problem Statement vs. Solution

### The Challenge
* **Input:** Geographic Coordinates (Lat/Lon).
* **Goal:** Verify installation and estimate power capacity.
* **The Trap:** Standard **Object Detection (Bounding Boxes)** is inaccurate for area estimation. Solar panels are often tilted or rotated; a rectangular box includes "empty" roof space, leading to overestimated capacity.

### Our Solution: Instance Segmentation
We utilize **YOLOv11-Large (Segmentation)** to predict **Polygons (Masks)** instead of boxes.
* **Precision:** Captures the exact shape of the panel array.
* **Math:** `Pixel Area` is calculated from the polygon mask, excluding background noise.
* **Output:** Verifiable JSON record with `panel_count`, `area_sqm`, and `capacity_kw`.

---

## 🛠️ Tech Stack

* **Core Framework:** [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) (PyTorch)
* **Model Architecture:** `yolo11l-seg.pt` (Large Segment Model)
* **Data Engineering:** [Roboflow](https://roboflow.com) (Annotation, Merging, Augmentation)
* **Compute:** Local GPU (NVIDIA T4/RTX) via VS Code & Google Colab.

---

## 📊 Data Engineering Strategy

To ensure robustness across India's diverse landscapes, we constructed a **"Super Dataset"** (~12,000 images) by merging three distinct sources to fix the Domain Gap.

| Source ID | Origin | Type | Purpose |
| :--- | :--- | :--- | :--- |
| **Source 1** | Google Maps | Screenshots | **Target Domain Test Data.** (Manually converted from Boxes to Polygons using SAM). |
| **Source 2** | LSGI547 | Drone Imagery | High-quality training data for feature extraction. |
| **Source 3** | Other | Drone Imagery | Supplementary training data. |

### Preprocessing & Augmentation
* **Resize:** `640x640` (Matched to YOLOv11 input).
* **Augmentations:**
    * **Flip:** Horizontal & Vertical (Orientation Invariance).
    * **Rotation:** ±15° (To handle tilted roofs).
    * **Blur:** Gaussian 1.5px (CRITICAL: Adapts high-res drone model to blurry satellite data).
    * **Generation:** 3x Multiplier (Expanded 4k raw images to ~12k training images).

---

## ⚙️ Installation

```bash
# Install core dependencies
pip install ultralytics roboflow opencv-python numpy