# Robust Offline Signature Verification Using Hu and Zernike Moments

## Overview

This project is an offline handwritten signature verification system that classifies signatures as **Genuine** or **Forged** using image processing and machine learning techniques. The system extracts discriminative shape and texture features from signature images and uses a trained Random Forest model to perform verification.

The application includes a Streamlit-based web interface that allows users to upload signature images and instantly receive verification results with confidence scores.

## Features

* Offline signature verification
* Signature image preprocessing
* Hu Moment feature extraction
* Zernike Moment feature extraction
* Feature fusion for improved accuracy
* Random Forest based classification
* Genuine/Forged prediction
* Confidence score display
* Streamlit web application

## Technology Stack

### Programming Language

* Python

### Libraries

* OpenCV
* NumPy
* Scikit-Learn
* Scikit-Image
* Mahotas
* Streamlit
* Pillow
* Matplotlib
* Joblib

## Dataset

The dataset used for training is **not included in this repository** due to licensing and distribution restrictions.

This project was trained using the **GPDS Signature Database (GPDS 1-150)**, a benchmark dataset widely used in offline signature verification research.

### Dataset Structure

```text
dataset/
└── train/
    ├── genuine/
    └── forge/
```

* `genuine/` contains authentic signature samples.
* `forge/` contains forged signature samples.

### Dataset Reference

Download the dataset used for this project [from here](https://www.kaggle.com/datasets/adeelajmal/gpds-1150?select=New+folder+%2810%29).

## Methodology

### 1. Image Preprocessing

Each signature image undergoes:

* Grayscale conversion
* Thresholding
* Noise removal
* Edge enhancement
* Image normalization
* Skeletonization (optional)

### 2. Feature Extraction

#### Hu Moments

Hu Moments provide seven invariant descriptors that capture the global geometric characteristics of a signature while remaining invariant to rotation, scaling, and translation.

#### Zernike Moments

Zernike Moments capture local structural details and shape information while maintaining rotational invariance.

### 3. Feature Fusion

The extracted Hu and Zernike features are normalized and combined into a single feature vector representing each signature.

### 4. Classification

The fused feature vector is passed to a trained Random Forest classifier which determines whether the signature is genuine or forged.

## System Workflow

```text
Input Signature
       │
       ▼
Preprocessing
       │
       ▼
Feature Extraction
(Hu + Zernike)
       │
       ▼
Feature Fusion
       │
       ▼
Random Forest Classifier
       │
       ▼
Genuine / Forged
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/AmanAdusumilli/signature-verification-using-a-fusion-of-Hu-and-Zernike-moments.git
cd signature-verification-using-a-fusion-of-Hu-and-Zernike-moments
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### create the .joblib file

```bash
python model.py
```

### Run the Application

```bash
streamlit run app.py
```

## Project Structure

```text
signature-verification/
│
├── dataset/
│   └── train/
│       ├── genuine/
│       └── forge/
│
├── app.py
├── model.py
├── requirements.txt
└── README.md
```

## Authors

**Aman Adusumilli**
B.Tech Computer Science and Engineering (AI & ML)
Mahatma Gandhi Institute of Technology

**B. Saaketh**
B.Tech Computer Science and Engineering (AI & ML)
Mahatma Gandhi Institute of Technology
