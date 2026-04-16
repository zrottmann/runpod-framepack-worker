#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh — FramePack I2V RunPod Serverless Worker
#
# Runs as PID 1 inside the container.  Two responsibilities:
#   1. Ensure the model directory tree exists on the network volume.
#      handler.py will populate these dirs on first invocation; subsequent
#      runs skip the download entirely.  mkdir -p is safe to call every boot.
#   2. Hand off to handler.py via `exec` (replaces shell PID, proper signal
#      forwarding to Python process).
#
# The -e flag aborts immediately if any command returns non-zero so that
# a misconfigured volume mount causes a loud failure instead of silent drift.
# =============================================================================
set -e

# ---------------------------------------------------------------------------
# Create the ComfyUI-style model subdirectories that handler.py expects to
# find weights in.  These paths mirror the ComfyUI model folder conventions
# used by FramePackWrapper (diffusion_models, text_encoders, vae, clip_vision).
# If /runpod-volume is a freshly attached empty volume this is a no-op for
# subsequent boots once the dirs already exist.
# ---------------------------------------------------------------------------
mkdir -p /runpod-volume/models/diffusion_models
mkdir -p /runpod-volume/models/text_encoders
mkdir -p /runpod-volume/models/vae
mkdir -p /runpod-volume/models/clip_vision

# ---------------------------------------------------------------------------
# Optional: symlink the volume model dirs into the FramePackWrapper's
# expected ComfyUI model paths so that any ComfyUI-aware loader that reads
# folder_paths also works without reconfiguration.
# ---------------------------------------------------------------------------
COMFY_MODELS_DIR="/app/framepack/models"
mkdir -p "$COMFY_MODELS_DIR"
for subdir in diffusion_models text_encoders vae clip_vision; do
    target="$COMFY_MODELS_DIR/$subdir"
    if [ ! -e "$target" ]; then
        ln -s "/runpod-volume/models/$subdir" "$target"
    fi
done

# ---------------------------------------------------------------------------
# Start the RunPod serverless worker.  `exec` replaces this shell so that
# SIGTERM/SIGINT from the RunPod platform are delivered directly to Python.
# -u forces unbuffered stdout/stderr so RunPod log streaming shows output
# immediately rather than after the buffer fills.
# ---------------------------------------------------------------------------
exec python -u /app/handler.py
