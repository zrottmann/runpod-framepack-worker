#!/bin/bash
set -eu

# Create ComfyUI-style model subdirectories on the network volume.
mkdir -p /runpod-volume/models/diffusion_models \
         /runpod-volume/models/text_encoders \
         /runpod-volume/models/clip \
         /runpod-volume/models/vae \
         /runpod-volume/models/clip_vision

# Symlink ComfyUI's model dirs to the volume so ComfyUI resolves weights
# at the standard /app/ComfyUI/models/<subdir> paths.
for sub in diffusion_models text_encoders clip vae clip_vision; do
    rm -rf "/app/ComfyUI/models/$sub"
    ln -s "/runpod-volume/models/$sub" "/app/ComfyUI/models/$sub"
done

# Handler is responsible for downloading models and starting ComfyUI.
exec python -u /app/handler.py
