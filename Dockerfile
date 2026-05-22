# ==========================================
# FastAPI Backend — Render Deployment
# ==========================================
FROM python:3.11-slim

# Prevent python from buffering stdout and pyc writing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for some Python packages (e.g. sentence-transformers)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using Uvicorn
# Uses PORT env var (set by Render) or defaults to 8000 for local dev
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
