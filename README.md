# Solar Panel Buffer Detection

A computer vision pipeline to detect solar panels and calculate their area within specific circular buffer zones (1200 Sq.Ft and 2400 Sq.Ft) using YOLOv11 and Shapely.

## Project Structure

```
├── main.py              # Main CLI entry point
├── requirements.txt     # Dependencies
├── src/
│   ├── geometry.py      # Buffer zone calculation logic
│   └── predictor.py     # YOLO inference wrapper
└── notebooks/           # Experiments
```

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ASHIF-MOHAMED/EcoInnovators-Solar-PV-Detection
   cd Ideathon-Solar-Detection
   ```

2. **Install Dependencies**
   It is recommended to use a virtual environment.
   ```bash
   # Create virtual env
   python -m venv venv
   # Activate (Windows)
   .\venv\Scripts\activate
   # Activate (Linux/Mac)
   source venv/bin/activate
   
   # Install libraries
   pip install -r requirements.txt
   ```

## Usage

Run the main script with your model and image:

```bash
python main.py --image test_imgs/test1.jpg --model TRAINED_MODEL/detection_model.pt
```

## Contributing

1. **Dependency Management**: We use `requirements.txt`. If you add a new library, please update `requirements.txt` with the exact version number (e.g., `pandas==2.1.0`) to ensure consistency for all users.
2. **Buffer Logic**: Core math logic is located in `src/geometry.py`.
