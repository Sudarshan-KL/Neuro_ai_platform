# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for MNE and scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libhdf5-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a venv so we can copy only what we need
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# Install CPU-only PyTorch first (saves ~800 MB vs default CUDA build)
RUN pip install --upgrade pip && \
    pip install torch==2.1.0+cpu torchvision==0.16.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Shared libraries required at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code
COPY . .

# Create runtime directories
RUN mkdir -p data saved_models logs

# Non-root user for security
RUN useradd -m -u 1000 neuro && chown -R neuro:neuro /app
USER neuro

# Expose a default port (not critical on Railway but ok to keep)
EXPOSE 8000

# Health check uses PORT env if present, else 8000 for local dev
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1

# Use Railway's PORT env var if set, fallback 8000 locally
CMD ["sh", "-c", "uvicorn app.main:app \
  --host 0.0.0.0 \
  --port ${PORT:-8000} \
  --workers 2 \
  --log-level info \
  --access-log"]
