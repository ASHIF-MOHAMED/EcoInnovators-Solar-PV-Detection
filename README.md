# Project Surya-Drishti: AI-Powered Solar Panel Verification System

An automated computer vision pipeline for remote verification of rooftop solar installations under India's **PM Surya Ghar: Muft Bijli Yojana**. The system uses satellite imagery to detect, quantify, and verify solar panels at given geographic coordinates, replacing manual field inspections.

## 🎯 Problem Statement

Traditional solar panel verification requires manual site visits, which are:
- Time-consuming and expensive
- Difficult to scale for nationwide programs
- Subject to human error in measurements

**Solution:** AI-powered remote verification using satellite imagery with instance segmentation for precise area calculation.

---

## 🏗️ Architecture

### Project Structure
```
Ideathon/
├── main.py                      # Batch processing pipeline (production)
├── requirements.txt             # Python dependencies
├── input_samples.xlsx           # Input template (sample_id, lat, lon)
├── TRAINED_MODEL/
│   └── detection_model.pt       # YOLOv11-Large segmentation model
├── src/
│   ├── predictor.py            # YOLO inference wrapper
│   ├── image_loader.py         # Satellite tile fetching (Esri)
│   ├── quality_checker.py      # Image quality control
│   ├── geometry.py             # Buffer zone analysis & area calculation
│   └── visualizer.py           # Result visualization
└── batch_output/               # Generated outputs (JSON + visualizations)
```

### Processing Pipeline

```
Input (Lat/Lon) → Satellite Image Fetch → Quality Check → Solar Detection
                                              ↓                   ↓
                                         NOT_VERIFIABLE    YOLO Segmentation
                                              ↓                   ↓
                                         Exit w/ JSON      Buffer Analysis
                                                                  ↓
                                                           JSON + Visualization
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/ASHIF-MOHAMED/EcoInnovators-Solar-PV-Detection
cd EcoInnovators-Solar-PV-Detection

# Create virtual environment (recommended)
python -m venv venv

# Activate environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Model

Ensure `TRAINED_MODEL/detection_model.pt` exists (119 MB, tracked via Git LFS).

---

## 📝 Usage

### Batch Processing

Process multiple sites from Excel/CSV input file:

```bash
python main.py -i input_samples.xlsx
```

**Input Format (Excel/CSV):**
| sample_id | lat      | lon      |
|-----------|----------|----------|
| 1001      | 9.8828   | 78.0836  |
| 1002      | 12.9906  | 80.2444  |
| 1003      | 13.0827  | 80.2707  |

**Outputs (per sample):**
- `batch_output/output_{sample_id}.json` - Detection results
- `batch_output/viz_{sample_id}.jpg` - Annotated visualization

**Advanced Options:**
```bash
# Custom model path
python main.py -i samples.csv -m path/to/model.pt

# Custom output directory
python main.py -i samples.xlsx -o results/

# Adjust fetch radius (default: 50m)
python main.py -i samples.xlsx -r 75
```

---

## 📊 Output Format

### JSON Structure

```json
{
  "sample_id": 1001,
  "lat": 9.8828,
  "lon": 78.0836,
  "has_solar": true,
  "confidence": 0.87,
  "pv_area_sqm_est": 26.22,
  "buffer_radius_sqft": 1200,
  "qc_status": "VERIFIABLE",
  "bbox_or_mask": "polygon_bounds_39x39",
  "image_metadata": {
    "source": "Google Maps",
    "capture_date": "2025-12-14"
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | int | Unique site identifier |
| `lat` / `lon` | float | WGS84 coordinates (4 decimal precision) |
| `has_solar` | bool | Solar panels detected in buffer zones |
| `confidence` | float | Average model confidence (0-1) |
| `pv_area_sqm_est` | float | Total panel area in square meters |
| `buffer_radius_sqft` | int | Detection zone (1200 or 2400 sq.ft) |
| `qc_status` | string | `VERIFIABLE` or `NOT_VERIFIABLE` |
| `bbox_or_mask` | string | Polygon/mask information for audit |
| `image_metadata` | object | Source and capture date |

### QC Status Codes

**VERIFIABLE:**
- Image quality acceptable
- Clear evidence of solar presence/absence

**NOT_VERIFIABLE:**
- High cloud coverage (>30%)
- Low resolution/blur
- Poor lighting (too dark/bright)
- Insufficient contrast

---

## 🔧 Technical Details

### Model Specifications
- **Architecture:** YOLOv11-Large (Instance Segmentation)
- **Training Data:** ~12,000 images (satellite + drone imagery)
- **Input Size:** 640x640 pixels
- **Confidence Threshold:** 0.15
- **Output:** Polygon masks (precise area calculation)

### Buffer Zone Logic
- **Buffer 1:** 1200 sq.ft radius (~19.6m) - High confidence detection
- **Buffer 2:** 2400 sq.ft radius (~27.7m) - Expanded search area
- Panels detected in Buffer 1 get priority classification

### Satellite Imagery
- **Source:** Esri World Imagery (ArcGIS tile service)
- **Zoom Level:** 19 (dynamic calculation, ~0.15 m/px)
- **Tile Grid:** 3×3 tiles (768×768 px stitched)
- **Crop Radius:** 50m around target coordinates

### Quality Control Checks
1. **Cloud Coverage:** < 30% white/bright pixels
2. **Blur Detection:** Laplacian variance > 100
3. **Brightness:** Range 30-240 (0-255 scale)
4. **Contrast:** Standard deviation > 20

---

## 🧪 Testing

```bash
# Test with sample file
python main.py -i input_samples.xlsx
```

---

## 📦 Dependencies

Core libraries (see `requirements.txt` for versions):
- **ultralytics** - YOLOv11 framework
- **opencv-python-headless** - Image processing
- **shapely** - Geometric operations
- **pyproj** - Coordinate transformations
- **pandas** - Data handling
- **requests** - HTTP tile fetching
- **pillow** - Image I/O

---

## 🤝 Contributing

1. **Code Style:** Follow PEP 8 conventions
2. **Dependencies:** Update `requirements.txt` with exact versions
3. **Testing:** Test `main.py` with sample data before commits
4. **Model Updates:** Use Git LFS for model files (`*.pt`)

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙋 Support

For issues or questions:
- Open an issue on GitHub
- Check existing documentation in `NON_VERIFIABLE_IMAGES_GUIDE.md`

---

## 🏆 Acknowledgments

Developed for **EcoInnovators Ideathon 2026** - PM Surya Ghar: Muft Bijli Yojana Challenge

**Team:** EcoInnovators  
**Project:** Surya-Drishti (Solar Vision)  
**Goal:** Enable scalable, accurate remote verification of residential solar installations across India
