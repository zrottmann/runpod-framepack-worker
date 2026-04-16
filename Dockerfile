# =============================================================================
# Dockerfile — FramePack I2V RunPod Serverless Worker
#
# Design: lightweight image (~2 GB) that ships code only.
# All model weights (~25 GB) live on a RunPod Network Volume mounted at
# /runpod-volume and are downloaded by handler.py on first invocation.
#
# Base image: runtime-only CUDA 12.1.1 on Ubuntu 22.04 (NOT devel, to keep
# the image small — we don't need nvcc or headers at runtime).
# =============================================================================

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ---------------------------------------------------------------------------
# Avoid interactive prompts from apt-get during the build.
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System deps:
#   - python3.10 + pip (Ubuntu 22.04 ships 3.10 as the default python3)
#   - ffmpeg   — for video encoding in imageio[ffmpeg]
#   - git      — for cloning ComfyUI-FramePackWrapper
#   - curl     — for health checks / HF downloads in handler fallback
#   - ca-certificates — needed for HTTPS HF downloads
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        ffmpeg \
        git \
        curl \
        ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Upgrade pip before installing anything else.
# ---------------------------------------------------------------------------
RUN python -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# PyTorch 2.4.0 + torchvision 0.19.0, CUDA 12.1 builds.
# Installed first so subsequent packages don't pull the CPU-only wheel.
# Using the official PyTorch index (download.pytorch.org/whl/cu121).
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        torch==2.4.0+cu121 \
        torchvision==0.19.0+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# Clone Kijai's ComfyUI-FramePackWrapper into /app/framepack.
# We do NOT run a ComfyUI server — we import the Python modules directly.
# The clone happens at build time so the image ships all Python source code.
# Pin to main; swap to a commit SHA for reproducibility once stable.
# ---------------------------------------------------------------------------
RUN mkdir -p /app && \
    git clone --depth 1 \
        https://github.com/kijai/ComfyUI-FramePackWrapper.git \
        /app/framepack

# ---------------------------------------------------------------------------
# Install FramePackWrapper's own Python dependencies.
# Its requirements.txt (if present) typically includes einops, diffusers etc.
# We install it here alongside our own requirements.txt below.
# ---------------------------------------------------------------------------
RUN if [ -f /app/framepack/requirements.txt ]; then \
        pip install --no-cache-dir -r /app/framepack/requirements.txt; \
    fi

# ---------------------------------------------------------------------------
# Project requirements (pinned versions — see requirements.txt for rationale).
# ---------------------------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---------------------------------------------------------------------------
# Copy application code.
# entrypoint.sh — creates volume dirs, then exec's handler.py
# handler.py    — the RunPod serverless handler
# ---------------------------------------------------------------------------
COPY handler.py     /app/handler.py
COPY entrypoint.sh  /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ---------------------------------------------------------------------------
# Add /app/framepack to PYTHONPATH so handler.py can do:
#   from nodes_framepack import FramePackSampler, ...
# without modifying sys.path at runtime.
# ---------------------------------------------------------------------------
ENV PYTHONPATH="/app/framepack:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# RunPod network volume will be mounted here by the platform at pod start.
# The directory is created here only as a fallback if the volume is absent.
# ---------------------------------------------------------------------------
RUN mkdir -p /runpod-volume/models

# ---------------------------------------------------------------------------
# Standard RunPod Serverless entrypoint.
# ---------------------------------------------------------------------------
ENTRYPOINT ["bash", "/app/entrypoint.sh"]
