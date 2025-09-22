# IOT-based Medical Plant Detection System

This project is an IoT-based system for detecting and reporting medical plants using computer vision (YOLO), MongoDB, and email notifications. It features a Flask web interface, real-time video, and integration with Arduino for hardware control.

## Features
- Real-time plant detection using YOLO
- MongoDB storage for detections and images
- Email notifications with detection reports
- Arduino hardware integration
- Web dashboard for viewing and managing detections

## Setup
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your sensitive configuration (see `.env` example)
4. Run the app:
   ```
   python app.py
   ```

## Folder Structure
- `app.py` - Main application file
- `static/` - Captured images (ignored by git)
- `templates/` - HTML templates
- `.env` - Environment variables (ignored by git)


