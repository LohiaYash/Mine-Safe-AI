# MineSafe-AI ⛏️

**Predict Rockfall Before It Happens.**

MineSafe-AI is an advanced, AI-powered rockfall hazard prediction system designed specifically for Indian open-pit mines. It integrates geological, environmental, and operational data to assess risk in real-time, protecting miners with actionable, data-driven safety intelligence.

## ✨ Key Features

- **Real-Time Risk Predictor:** An interactive web tool that analyzes 14+ specific parameters (e.g., Slope Angle, Weathering Grade, Joint Friction) to instantly calculate a composite **Risk Score** and determine the overall **Hazard Level**.
- **YOLO Vision Detection:** A robust computer vision pipeline (powered by YOLOv8) to detect rockfall events in real-time from slope surveillance camera feeds.
- **Dynamic Analytics Dashboard:** Live visual tracking of risk distributions, feature importance (via correlation), regional data profiles, and model performance metrics, utilizing `Chart.js`.
- **Context-Aware Recommendations:** Automatically suggests safety and mitigation measures based on the specific hazard triggers detected (e.g., heavy rainfall or excessive vibrations).
- **Indian Mining Context:** Specialized profiles and risk analyses tailored to the varying topographies of Indian mining corridors including Jharkhand (Coal), Odisha (Iron Ore), and Rajasthan (Limestone).

## 🚀 Tech Stack

- **Frontend:** HTML5, CSS3 (Premium Dark Theme, Glassmorphism), Vanilla JavaScript, Chart.js
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn (Random Forest, Gradient Boosting), Pandas, NumPy
- **Computer Vision:** Ultralytics YOLOv8, OpenCV

## 📁 Core Project Structure

```text
MineSafe-AI/
├── app.py                  # Flask backend API and web server
├── main.py                 # Main CLI for Model Training and Batch Prediction
├── infer_video.py          # YOLO object detection module for video feeds
├── config.yaml             # System & ML configuration
├── data/
│   └── datasets/           # Mining site datasets used for training and stats
├── src/
│   ├── predictor.py        # ML core prediction system
│   └── data_loader.py      # Data preprocessing and feature engineering
├── static/
│   ├── css/style.css       # Frontend premium design system
│   └── js/app.js           # Frontend interactions, sliders, and charting logic
└── templates/
    └── index.html          # The single-page interactive application
```

## ⚙️ Quick Start

**1. Install Dependencies:**
Ensure Python is installed, then run:
```bash
pip install flask pandas numpy pyyaml scikit-learn ultralytics opencv-python
```

**2. Start the Web Server:**
```bash
python3 app.py
```
*(By default, the application runs on `http://127.0.0.1:5050`)*

**3. Explore the Dashboard:**
Open your browser and navigate to the address above to access the animated hero, interactive risk predictor, and intelligent dashboard.

## 🧠 Model Training

To train or retrain the machine learning models on new dataset files:
```bash
python3 main.py --mode train
```

---
*Built to improve mining safety and protect lives through comprehensive AI.*
