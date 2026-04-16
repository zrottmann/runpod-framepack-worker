FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps: Python, ffmpeg, git, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        git \
        curl \
        ca-certificates \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# PyTorch cu121 — install first so later packages don't pull CPU wheels
RUN pip install --no-cache-dir \
        torch==2.4.0 \
        torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu121

# ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /app/ComfyUI
RUN pip install --no-cache-dir -r /app/ComfyUI/requirements.txt

# Custom nodes: FramePackWrapper + VideoHelperSuite
RUN cd /app/ComfyUI/custom_nodes \
    && git clone --depth 1 https://github.com/kijai/ComfyUI-FramePackWrapper \
    && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite \
    && for d in ComfyUI-FramePackWrapper ComfyUI-VideoHelperSuite; do \
         if [ -f "$d/requirements.txt" ]; then \
           pip install --no-cache-dir -r "$d/requirements.txt"; \
         fi \
       ; done

# Handler requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Application code + workflow
COPY handler.py      /app/handler.py
COPY entrypoint.sh   /app/entrypoint.sh
COPY framepack_long_i2v_remote.json /app/workflow.json
RUN chmod +x /app/entrypoint.sh

# Fallback volume root (real volume mounted at runtime by RunPod)
RUN mkdir -p /runpod-volume/models

ENTRYPOINT ["/app/entrypoint.sh"]
