# Pneumonia Detection using Deep Learning

## Overview
This project is a Deep Learning-based Pneumonia Detection system that predicts whether a patient has pneumonia using chest X-ray images. The model is trained using Convolutional Neural Networks (CNN) and provides accurate classification results between Normal and Pneumonia cases.

---

## Features
- Chest X-ray image classification
- Pneumonia detection using CNN
- Image upload and prediction system
- Deep Learning-based medical diagnosis support
- Simple and user-friendly interface

---

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Flask
- HTML/CSS

---

## Dataset Structure

```bash
dataset/
│
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── valid/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Project Structure

```bash
PneumoniaDetection/
│
├── dataset/
│   ├── test/
│   ├── train/
│   └── valid/
│
├── static/
│   ├── css/
│   ├── images/
│   └── uploads/
│
├── templates/
│
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/PneumoniaDetection.git
cd PneumoniaDetection
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python app.py
```

Open your browser and visit:

```bash
http://127.0.0.1:5000
```

---

## Model Training

```bash
python train_model.py
```

---

## Prediction

Upload a chest X-ray image through the web interface to predict:
- NORMAL
- PNEUMONIA
- UNKNOWN

---

## Applications
- Medical image analysis
- AI-assisted healthcare
- Early pneumonia diagnosis
- Hospital support systems

---

## Future Improvements
- Improve model accuracy
- Add multi-disease detection
- Deploy using cloud platforms
- Real-time prediction support

---

## Author
**Manya B.J**

---

## License
This project is intended for educational and research purposes only.
