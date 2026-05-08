# SignX 🤟

> **Text-to-3D Motion Generation for American Sign Language (ASL)**

SignX is an end-to-end deep learning system that translates natural English text into continuous, culturally authentic 3D ASL animations. Instead of pre-recorded video clips, SignX generates raw skeletal motion — frame by frame — making sign language accessibility scalable, automated, and dynamic.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [ML Pipeline](#ml-pipeline)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Roadmap](#roadmap)

---

## Overview

Over **430 million** deaf and hard-of-hearing individuals worldwide rely on sign language as their primary language. Yet most digital content offers only written text or closed captions — effectively a second language for native signers.

SignX solves this at scale:

- ✅ No studio recording required
- ✅ Works on dynamic, real-time content
- ✅ Generates expressive body, hand, and facial motion
- ✅ Outputs renderable 3D skeletal coordinates (Mediapipe format)

---

## Architecture

SignX is a three-tier system:

```
┌─────────────────────────────────────────────────────┐
│                   FRONTEND (React)                  │
│  Text Input  →  WebGL 3D Viewer  →  MP4 Export      │
└───────────────────────┬─────────────────────────────┘
                        │ HTTP (JSON)
┌───────────────────────▼─────────────────────────────┐
│               BACKEND (FastAPI)                     │
│  Request Routing  →  Coordinate Smoothing           │
└───────────────────────┬─────────────────────────────┘
                        │ PyTorch Inference
┌───────────────────────▼─────────────────────────────┐
│             ML PIPELINE (PyTorch)                   │
│  DistilBERT → Length Net → Transformer Decoder      │
└─────────────────────────────────────────────────────┘
```

### Frontend
Built with **React** and **React Three Fiber** (Three.js). Renders Mediapipe landmark outputs onto a skeletal rig in real-time via a WebGL canvas. Supports playback speed control and MP4 export.

### Backend
A **FastAPI** server handles text payloads, routes them through the PyTorch inference pipeline, smooths raw coordinates, and returns a structured JSON response the frontend can consume directly.

### ML Pipeline
The core neural engine — see [ML Pipeline](#ml-pipeline) below.

---

## ML Pipeline

The translation happens in four sequential phases:

| Phase | Component | Description |
|-------|-----------|-------------|
| 1 | **Semantic Encoder** | DistilBERT converts input text into a dense contextual embedding |
| 2 | **Length Estimator** | Neural network predicts the required frame count for the sequence |
| 3 | **Motion Decoder** | Spatial-Temporal Transformer predicts (X, Y, Z) coordinates per frame |
| 4 | **Optimization** | MSE loss against How2Sign ground-truth motion capture data |

The decoder generates coordinates for **body**, **hands**, and **face** landmarks simultaneously, preserving the spatial grammar of ASL.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React, React Three Fiber, Three.js |
| Backend | Python, FastAPI |
| ML Framework | PyTorch |
| Language Model | DistilBERT (HuggingFace Transformers) |
| Pose Estimation | Google Mediapipe (Holistic) |
| Dataset | How2Sign Holistic 3D |

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- CUDA-capable GPU (recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/signx.git
cd signx

# Backend setup
cd backend
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
```

### Running the App

```bash
# Start the backend API
cd backend
uvicorn main:app --reload --port 8000

# Start the frontend (in a new terminal)
cd frontend
npm run dev
```

Then open `http://localhost:3000` in your browser.

### Running Inference

```python
from signx.pipeline import SignXPipeline

pipeline = SignXPipeline.from_pretrained("checkpoints/signx-base")
coordinates = pipeline.translate("Hello, how are you?")
# Returns: array of shape (T, N_landmarks, 3)
```

---

## Project Structure

```
signx/
├── backend/
│   ├── main.py               # FastAPI app entry point
│   ├── inference.py          # PyTorch model loading & inference
│   ├── coordinate_utils.py   # Smoothing & JSON formatting
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── TextInput.jsx
│   │   │   ├── SignViewer.jsx # Three.js / R3F canvas
│   │   │   └── ExportPanel.jsx
│   │   └── App.jsx
│   └── package.json
├── ml/
│   ├── model/
│   │   ├── encoder.py        # DistilBERT semantic encoder
│   │   ├── length_net.py     # Frame count predictor
│   │   └── transformer.py    # Spatial-Temporal Transformer Decoder
│   ├── train.py
│   ├── dataset.py            # How2Sign Holistic data loader
│   └── evaluate.py
├── checkpoints/              # Saved model weights (gitignored)
├── notebooks/                # Exploratory analysis & demos
└── README.md
```

---

## Dataset

SignX trains on the **[How2Sign](https://how2sign.github.io/)** dataset — a large-scale multimodal dataset of continuous American Sign Language, recorded with full holistic pose annotations via Google Mediapipe.

To set up the dataset:

```bash
# Download instructions at https://how2sign.github.io/
# Place extracted files in:
data/how2sign/
├── train/
├── val/
└── test/
```

---

## Roadmap

- [x] Model architecture design
- [x] How2Sign Holistic data pipeline
- [ ] Transformer decoder training (in progress)
- [ ] React frontend with Three.js skeletal renderer
- [ ] FastAPI backend integration
- [ ] Real-time speech-to-sign (audio → 3D motion)
- [ ] British Sign Language (BSL) support
- [ ] Sri Lankan Sign Language support
- [ ] Metaverse / avatar integration

---

<p align="center">Built to make the digital world a little more human.</p>
