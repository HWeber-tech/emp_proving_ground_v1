# EMP Ultimate Architecture v1.1 - Production Dockerfile
# Multi-stage build for optimized production deployment

# Stage 1: Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Set labels
LABEL maintainer="EMP System <emp@example.com>"
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.revision=$VCS_REF
LABEL org.opencontainers.image.version=$VERSION
LABEL org.opencontainers.image.title="EMP Ultimate Architecture v1.1"
LABEL org.opencontainers.image.description="Evolutionary Market Prediction System"
LABEL org.opencontainers.image.vendor="EMP System"

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
# Use the pinned requirements file instead of the main requirements to avoid
# installing unavailable or incompatible packages (e.g. the cTrader API).  The
# requirements-fixed.txt file contains version constraints tested for Python
# 3.8+ and does not include optional packages that require manual installation.
COPY requirements-fixed.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-fixed.txt

# Stage 2: Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r emp && useradd -r -g emp emp

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY main.py .
COPY run_genesis.py .

# Create necessary directories
RUN mkdir -p data logs reports && \
    chown -R emp:emp /app

# Switch to non-root user
USER emp

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV EMP_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "main.py"]
