# Base image: lightweight Python
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal, no ffmpeg)
RUN apt-get update && apt-get install -y \
    git \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
# Ultralytics YOLOv8 + Torch + OpenCV
RUN pip install ultralytics==8.2.0 torch==2.5.1 opencv-python-headless

# If you have requirements.txt, better to use:
# RUN pip install -r requirements.txt

# Expose port (for FastAPI/Flask if you add API later)
EXPOSE 8000

# Default command: run prediction test
CMD ["python", "test/test_prediction.py"]
