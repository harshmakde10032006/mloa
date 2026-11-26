# ==============================================================================
# Dockerfile: Iris Classification ML Model REST API Service
# Course: MLOps (Machine Learning Operations) - Unit III
# Base Image: Python 3.11 Slim (Minimalist, production-ready, lightweight Debian)
# ==============================================================================

# Step 1: Specify the official lightweight Python base image
FROM python:3.11-slim

# Step 2: Set environment variables for non-buffered stdout/stderr and Python bytecode optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Step 3: Establish working directory inside the container
WORKDIR /app

# Step 4: Leverage Docker layer caching by copying dependencies file first
COPY requirements.txt .

# Step 5: Install Python dependencies without caching wheels to minimize image layer size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 6: Copy pre-trained serialized model artifact into working directory
COPY model.joblib .

# Step 7: Copy application code into container
COPY app.py .

# Step 8: Document and expose application port (Flask default)
EXPOSE 5000

# Step 9: Configure default container entrypoint command to launch Flask server
CMD ["python", "app.py"]
