# ==========================================
# Stage 1: Build the React Frontend
# ==========================================
FROM node:20-alpine AS frontend-build
WORKDIR /app/ui

# Copy frontend package files
COPY ui/package*.json ./

# Install dependencies
RUN npm install

# Copy frontend source code and build
COPY ui/ ./
RUN npm run build


# ==========================================
# Stage 2: Build the Python FastAPI Backend
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

# Copy the built React app from Stage 1 into the backend's expected directory
COPY --from=frontend-build /app/ui/dist /app/ui/dist

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
