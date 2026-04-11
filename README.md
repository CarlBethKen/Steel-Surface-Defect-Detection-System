# Steel Surface Defect Detection System

A web-based steel surface defect detection application powered by YOLOv8 deep learning model.

## Features

- **Real-time Monitoring** — Live camera feed with real-time defect detection; automatically saves records when defects are found
- **Image Detection** — Upload single/batch images for defect detection; supports drag-and-drop and folder upload
- **History Records** — View and manage all detection records with batch delete and export
- **Statistical Analysis** — Total detections, defect distribution, average confidence and more

## Project Structure

```
├── app.py                  # FastAPI backend application
├── run.py                  # One-click launcher script
├── requirements.txt        # Python dependencies
├── core/
│   ├── infer.py           # YOLOv8 model inference
│   ├── preprocess.py      # Image preprocessing
│   ├── draw.py            # Detection result visualization
│   ├── database.py        # SQLite database operations
│   └── storage.py         # CSV backup storage
├── models/
│   └── best.pt            # YOLOv8 trained weights
├── templates/
│   └── system.html        # Frontend page
└── static/
    ├── uploads/           # Uploaded images (generated at runtime)
    └── results/           # Detection result images (generated at runtime)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Model Weights

Put the trained YOLOv8 weight file `best.pt` into the `models/` directory.

### 3. Launch

```bash
python run.py
```

The system will start the server and open the browser at http://localhost:8000/system

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Model**: YOLOv8 (ultralytics)
- **Database**: SQLite + SQLAlchemy
- **Image Processing**: OpenCV
- **Frontend**: HTML5 + CSS3 + JavaScript
