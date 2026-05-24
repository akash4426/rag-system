# ==========================================
# Stage 1: Build the React Frontend
# ==========================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/ui

# Copy package files and install dependencies (use npm ci for reproducible builds)
COPY ui/package*.json ./
RUN npm ci --prefer-offline --no-audit

# Copy the rest of the frontend source code and build it
COPY ui/ .
RUN npm run build

# ==========================================
# Stage 2: Build the FastAPI Backend & Serve
# ==========================================
FROM python:3.11-slim

# Prevent python from buffering stdout and pyc writing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    libssl-dev \
    libffi-dev \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . .

# Copy the built React app from Stage 1 into the backend container
COPY --from=frontend-builder /app/ui/dist /app/ui/dist

# Create logs directory
RUN mkdir -p /app/logs

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Production: Run with Gunicorn + Uvicorn workers
# For Render/Cloud: set PORT env var, defaults to 8000
CMD gunicorn api.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} --timeout 120
