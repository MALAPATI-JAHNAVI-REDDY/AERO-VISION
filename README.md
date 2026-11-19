# AeroVision 🚁

**AI-Powered Drone Vision & Surveillance System**

AeroVision is an intelligent computer vision system for drone-based visual monitoring, combining object detection, live camera monitoring, media analysis, and an interactive web dashboard.

## Features

- Real-time object detection using YOLOv5n.
- Live camera monitoring with detection overlays.
- Image and MP4 video analysis.
- Frame-by-frame video analysis.
- Object tracking and detection activity.
- Interactive Next.js dashboard.
- Raspberry Pi Camera Module 3 integration.

## Technology Stack

| Component | Technology |
|---|---|
| AI Model | YOLOv5n |
| Runtime | ONNX Runtime |
| Computer Vision | OpenCV |
| Backend | Python, FastAPI |
| Frontend | Next.js, React, TypeScript |
| Styling | Tailwind CSS |
| Hardware | Raspberry Pi Zero 2 W |

## Project Structure

- backend/ - FastAPI application and APIs
- frontend/ - Next.js dashboard
- models/ - YOLOv5 ONNX model
- pi-agent/ - Raspberry Pi camera and inference modules
- documentation/ - Development and testing utilities
- sample-images/ - Sample images

## Installation

### Prerequisites

- Python 3.12
- Node.js and npm
- Git

### Clone Repository

`powershell
git clone https://github.com/MALAPATI-JAHNAVI-REDDY/AERO-VISION.git
cd AERO-VISION
Backend Setup
Create and activate a virtual environment:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Install dependencies:
python -m pip install fastapi uvicorn python-multipart opencv-python numpy onnxruntime pydantic


Start the backend:
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload


Backend: http://127.0.0.1:8000
Frontend Setup
Open another PowerShell window:
cd frontend
npm install
npm run dev


Dashboard: http://localhost:3000
Raspberry Pi Integration
Designed for future integration with:
- Raspberry Pi Zero 2 W
- Raspberry Pi Camera Module 3
- YOLOv5n ONNX model
Project Scope
AeroVision focuses on visual monitoring and object detection. It does not directly control drone flight, navigation, or motors.
Development Status
Active development.
License
Not yet specified.

AeroVision — Giving drones the ability to see and understand their surroundings.
