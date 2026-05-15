# Multi-stage build for smaller final image
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy source and install
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir --prefix=/install .

# ============ Production stage ============
FROM python:3.12-slim
# Install runtime dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
libgles2 \
    libgl1 \
    libegl1 \
    libglib2.0-0 \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libv4l-0 \
    v4l-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code and model
COPY src/ ./src/
COPY hand_landmarker.task ./

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000
ENV MEDIAPIPE_DISABLE_GPU=1


# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/status || exit 1

# Entry point
CMD ["python", "-m", "hand_control"]
