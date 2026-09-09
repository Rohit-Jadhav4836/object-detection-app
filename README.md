# Real-Time Object Detection App 🎯

A production-ready object detection application built with **Streamlit** and **YOLOv8**, featuring live webcam detection, batch processing, and a unique retro terminal aesthetic.

[![Streamlit](https://img.shields.io/badge/Streamlit-1.62.0-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat)](https://github.com/ultralytics/ultralytics)

---

** Live Preview :- https://object-detection-app-745apafghmk7hcjpo8mgse.streamlit.app/

** Interface Preview :-  


<img width="1917" height="980" alt="Screenshot 2026-09-09 174549" src="https://github.com/user-attachments/assets/c6b166d7-fbea-4da4-863b-178d88acf917" />


<img width="1917" height="912" alt="Screenshot 2026-09-09 174939" src="https://github.com/user-attachments/assets/f764d503-882e-40ed-b2f0-926832b644a9" />



<img width="1917" height="902" alt="Screenshot 2026-09-09 175026" src="https://github.com/user-attachments/assets/f9b99711-2586-4893-82f0-93d3f66a355c" />



<img width="1916" height="921" alt="Screenshot 2026-09-09 175055" src="https://github.com/user-attachments/assets/61b9fc2b-8aed-4c42-a518-8ab65cd94b1a" />




## ✨ Key Features

### 🎬 Three Detection Modes

1. **Live Webcam** - Real-time object detection from your camera (10-30 FPS)
2. **Single Image Upload** - Analyze individual images with instant results
3. **Batch Processing** - Process multiple images simultaneously with bulk export

### 🎨 Advanced Visualization

- **Confidence-Based Color Coding**: Red (<60%), Yellow (60-80%), Green (>80%)
- **Dual Color Schemes**: Choose between class-based or confidence-based box colors
- **Retro Terminal UI**: Custom CRT-style interface with scanlines and glow effects
- **Quality Indicators**: Automatic warnings for low-confidence or overlapping detections

### ⚙️ Smart Controls

- **5 Model Sizes**: Choose from YOLOv8n to YOLOv8x (speed vs accuracy tradeoff)
- **Adjustable Thresholds**: Confidence (0.5 default) and IoU/NMS (0.5 default)
- **Class Filtering**: Select specific objects from 80 COCO classes
- **Scrollable Sidebar**: Easy access to all controls

### 📊 Export & Analysis

- Download annotated images (PNG)
- Export detection data (JSON with metadata)
- Batch export (ZIP archives + consolidated JSON)
- Detailed detection logs with confidence scores

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (optional, for live detection)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd object-detection-app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Open your browser to **http://localhost:8501**

> **Note**: On first run, YOLOv8 will automatically download model weights (~6MB). Ensure you have an internet connection.

---

## 📖 Usage Guide

### 1️⃣ Live Webcam Mode

1. Select "Live Webcam" in sidebar
2. Click START and grant camera permissions
3. View real-time detection with live object tracking
4. Detections update automatically every 0.5 seconds

### 2️⃣ Image Upload Mode

1. Select "Image Upload" in sidebar
2. Upload a JPG/PNG image
3. View side-by-side comparison with detection results
4. Download annotated image or JSON data

### 3️⃣ Batch Processing Mode

1. Check "📦 Batch Processing Mode"
2. Upload multiple images (10, 20, 50+)
3. View processing progress bar
4. Review individual results in expandable panels
5. Download all as ZIP + consolidated JSON

### ⚙️ Optimize Detection Quality

**Recommended Settings:**
- **Confidence Threshold**: 0.5-0.6 (reduces false positives)
- **IoU Threshold**: 0.5-0.6 (reduces overlapping boxes)
- **Model Size**: yolov8m or larger for better accuracy
- **Color Scheme**: Switch to "Confidence Level" to visually spot low-quality detections

**Class Filtering:**
- Open "🔍 Class Filter" in sidebar
- Select specific classes (e.g., person, car, dog)
- Only selected objects will be detected

---

## 🏗️ Project Structure

```
object-detection-app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── assets/
│   └── style.css              # Retro terminal CSS styling
├── src/
│   ├── detector.py            # YOLO detection logic
│   └── utils.py               # Helper functions
├── README.md                  # This file
├── DETECTION_ISSUES.md        # Model limitations guide
├── PYARROW_FIX.md             # Technical troubleshooting
└── LINKEDIN_POST.md           # Social media templates
```

---

## 🎯 Technical Highlights

### Model & Architecture

- **YOLOv8** by Ultralytics (state-of-the-art object detection)
- **COCO Dataset**: Pre-trained on 80 object classes
- **5 Model Variants**: n (nano) to x (extra-large)
- **Real-time Inference**: 10-30 FPS on CPU (yolov8n)

### Key Technologies

- **Streamlit**: Web app framework
- **OpenCV**: Image processing and annotation
- **streamlit-webrtc**: Real-time webcam streaming
- **Pillow**: Image handling
- **NumPy & Pandas**: Data processing

### Performance Optimizations

- Model caching with `@st.cache_resource`
- Efficient BGR→RGB conversion
- Thread-safe detection updates for webcam
- Batch processing with progress tracking

---

## 🎨 Features Deep Dive

### Confidence-Based Color Coding

Every bounding box includes a visual quality indicator:

```
ELEPHANT 91%  ← High confidence (green)
DOG 72%       ← Medium confidence (yellow)
CAT 45%       ← Low confidence (red)
```

### Automatic Quality Warnings

```
⚠️ Quality Alert: 5 detection(s) with low confidence (<60%)
   → Increase confidence threshold to 0.6+

⚠️ Overlapping Detections: Multiple boxes on same objects
   → Increase IoU threshold to 0.6+
```

### Batch Processing Results

```
Total Images: 25
Total Objects Detected: 143
Average Inference Time: 89ms
```

---

## 📊 Model Information

### COCO Dataset Classes (80 total)

**People & Animals:**
person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles:**
bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Indoor Objects:**
bottle, wine glass, cup, fork, knife, spoon, bowl, chair, couch, bed, dining table, toilet

**Electronics:**
tv, laptop, mouse, remote, keyboard, cell phone

**Sports & Outdoor:**
frisbee, skis, snowboard, sports ball, kite, baseball bat, skateboard, surfboard, tennis racket

See [COCO dataset](https://cocodataset.org/) for the complete list.

### Model Performance Benchmarks

| Model | Size | Speed (CPU) | mAP | Use Case |
|-------|------|-------------|-----|----------|
| YOLOv8n | 6 MB | 10-20 FPS | 37.3 | Real-time demos |
| YOLOv8s | 22 MB | 6-12 FPS | 44.9 | Balanced |
| YOLOv8m | 52 MB | 4-6 FPS | 50.2 | **Recommended** |
| YOLOv8l | 87 MB | 2-3 FPS | 52.9 | High accuracy |
| YOLOv8x | 136 MB | 1-2 FPS | 53.9 | Best accuracy |

*Benchmarks on standard CPU. GPU provides 5-10x speedup.*

---

## ⚠️ Known Limitations

### COCO Dataset Constraints

YOLOv8 is trained on **80 broad categories only**:

- ✅ Detects: person, dog, cat, bird, car, etc.
- ❌ Does NOT detect: specific breeds, exotic animals, fine-grained species

**Examples:**
- Leopard/Jaguar/Cheetah → classified as "cat"
- Red Fox → may be "cat" or "dog"
- Parrot/Eagle/Owl → all classified as "bird"

### Solutions

1. **Accept limitations** for general-purpose detection
2. **Use class filtering** to reduce noise
3. **Fine-tune custom model** for specialized needs

See [DETECTION_ISSUES.md](DETECTION_ISSUES.md) for detailed guide.

---

## 🐛 Troubleshooting

### Webcam Not Working

- Ensure browser has camera permissions enabled
- Try Chrome or Edge (recommended)
- Close device selection preview before clicking START
- Check if another app is using the camera
- Resolution set to 640×480 (fixes most issues)

### Low Detection Quality

- Increase **Confidence Threshold** to 0.6+
- Increase **IoU Threshold** to 0.6+
- Use larger model (yolov8m or yolov8l)
- Enable "Color by Confidence" to identify weak detections

### Model Download Fails

- Check internet connection
- Ensure write permissions in project directory
- Model is cached after first successful download

### Performance Issues

- Use smaller model (yolov8n) for speed
- Reduce image resolution for batch processing
- Close other applications to free CPU/RAM

---

## 🚀 Deployment

### Deploy to Streamlit Cloud (Free)

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io/)
3. Connect your repository
4. Deploy automatically

**Benefits:**
- HTTPS by default (camera works everywhere)
- Automatic dependency management
- Free hosting for public apps
