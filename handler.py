"""
handler.py — FramePack I2V RunPod Serverless Worker
=====================================================

RunPod Serverless contract
--------------------------
RunPod calls handler(job) once per request.  job["input"] carries the user
payload.  The function must return a plain dict (serialised to JSON by the
platform).  On error, return {"error": str, "traceback": str} — RunPod
surfaces this back to the caller unchanged.

Lazy-load strategy
------------------
Model loading (~30 s on first cold start) happens only on the first handler()
invocation, NOT at import time.  This keeps the container startup fast enough
for RunPod's liveness check.  Subsequent calls reuse the cached globals.

FramePackWrapper API notes
--------------------------
ComfyUI-FramePackWrapper (github.com/kijai/ComfyUI-FramePackWrapper) ships
two Python modules of interest:

  nodes_framepack.py   — defines ComfyUI nodes: FramePackSampler,
                         LoadFramePackModel, etc.  Each node exposes an
                         execute(**kwargs) method; we call those directly
                         without running a ComfyUI server.

  framepack/           — Kijai's vendored copy of lllyasviel's FramePack
                         source (src/diffusers_helper, etc.).

The ComfyUI node system uses folder_paths.get_filename_list() to resolve
model filenames.  We monkey-patch folder_paths so the nodes find weights at
our /runpod-volume paths without a full ComfyUI installation.

If the direct-import approach breaks (e.g. after a Kijai update tightens
ComfyUI coupling), see the FALLBACK block at the bottom of _generate() for
the headless-subprocess alternative.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
import runpod
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths — all model weights live on the RunPod network volume.
# These mirror the ComfyUI model subdirectory conventions so that
# FramePackWrapper's folder_paths calls resolve correctly after our patch.
# ---------------------------------------------------------------------------
VOLUME_ROOT = Path("/runpod-volume/models")
MODEL_DIRS = {
    "diffusion_models": VOLUME_ROOT / "diffusion_models",
    "text_encoders":    VOLUME_ROOT / "text_encoders",
    "vae":              VOLUME_ROOT / "vae",
    "clip_vision":      VOLUME_ROOT / "clip_vision",
}

# ---------------------------------------------------------------------------
# Model manifest — parsed verbatim from remote_models_framepack.yml.
# Each entry: (subdir, filename, url, expected_size_bytes).
# Size is used as a quick integrity check; re-download if actual < expected.
# ---------------------------------------------------------------------------
MODEL_MANIFEST = [
    {
        "subdir":    "diffusion_models",
        "filename":  "FramePackI2V_HY_fp8_e4m3fn.safetensors",
        "url":       "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/FramePackI2V_HY_fp8_e4m3fn.safetensors",
        "size_bytes": 16_331_849_976,  # x-linked-size confirmed 2025-04-16
    },
    {
        "subdir":    "text_encoders",
        "filename":  "llava_llama3_fp8_scaled.safetensors",
        "url":       "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors",
        "size_bytes": 9_091_392_483,
    },
    {
        "subdir":    "text_encoders",
        "filename":  "clip_l.safetensors",
        "url":       "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/clip_l.safetensors",
        "size_bytes": 246_144_152,
    },
    {
        "subdir":    "vae",
        "filename":  "hunyuan_video_vae_bf16.safetensors",
        "url":       "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/hunyuan_video_vae_bf16.safetensors",
        "size_bytes": 492_986_478,
    },
    {
        "subdir":    "clip_vision",
        "filename":  "sigclip_vision_patch14_384.safetensors",
        "url":       "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors",
        "size_bytes": 856_505_640,
    },
]

# ---------------------------------------------------------------------------
# Global model cache — populated once on first handler() call.
# Keys match what the FramePackWrapper nodes return from their execute()
# methods.  All are None until ensure_pipeline_loaded() runs.
# ---------------------------------------------------------------------------
_PIPELINE: dict[str, Any] = {
    "framepack_model": None,   # transformer + scheduler
    "vae":             None,
    "clip":            None,   # dual CLIP (llava + clip_l)
    "clip_vision":     None,   # SigCLIP image encoder
}
_PIPELINE_LOADED = False


# =============================================================================
# Model download helpers
# =============================================================================

def _needs_download(entry: dict) -> bool:
    """Return True if the file is missing or smaller than expected."""
    dest = MODEL_DIRS[entry["subdir"]] / entry["filename"]
    if not dest.exists():
        return True
    actual = dest.stat().st_size
    # Allow 0.1 % tolerance for edge-case filesystem reporting differences.
    tolerance = entry["size_bytes"] * 0.001
    return actual < (entry["size_bytes"] - tolerance)


def _download_one(entry: dict) -> str:
    """
    Stream-download a single model file from HuggingFace into the volume.
    Uses HF_TOKEN env var if present (required for gated repos).
    Returns the filename on success; raises on any HTTP / IO error.
    """
    dest_dir = MODEL_DIRS[entry["subdir"]]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / entry["filename"]

    headers = {}
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    print(f"[download] Starting: {entry['filename']}", flush=True)
    with requests.get(entry["url"], headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or entry["size_bytes"]
        with open(dest, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=entry["filename"],
            leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"[download] Done: {entry['filename']}", flush=True)
    return entry["filename"]


def ensure_models_downloaded(job: dict) -> None:
    """
    Check every model file; concurrently download any that are missing or
    incomplete.  Uses a ThreadPoolExecutor of 5 workers so all files can
    download in parallel (HF CDN handles concurrent connections fine).
    """
    to_download = [e for e in MODEL_MANIFEST if _needs_download(e)]
    if not to_download:
        print("[models] All model files verified on volume.", flush=True)
        return

    print(
        f"[models] {len(to_download)} file(s) need downloading "
        f"({sum(e['size_bytes'] for e in to_download) / 1e9:.1f} GB total).",
        flush=True,
    )
    runpod.serverless.progress_update(job, {"stage": "downloading_models", "count": len(to_download)})

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_download_one, e): e for e in to_download}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                future.result()
            except Exception as exc:
                # Re-raise so the handler returns an error dict rather than
                # silently proceeding with a corrupt or missing weight file.
                raise RuntimeError(
                    f"Failed to download {entry['filename']}: {exc}"
                ) from exc


# =============================================================================
# FramePackWrapper monkey-patch for folder_paths
# =============================================================================

def _patch_folder_paths() -> None:
    """
    ComfyUI-FramePackWrapper imports `folder_paths` from ComfyUI's core to
    locate model files.  We don't run ComfyUI, so we inject a minimal shim
    module into sys.modules before importing the wrapper nodes.

    The shim maps each model category to our /runpod-volume/models/* dirs
    and provides the two functions the nodes actually call:
      - get_filename_list(folder_name)  -> list[str]  (filenames only)
      - get_full_path(folder_name, filename) -> str    (absolute path)
    """
    import types

    shim = types.ModuleType("folder_paths")

    # Map ComfyUI folder names to our volume paths.
    _FOLDER_MAP: dict[str, Path] = {
        "diffusion_models": MODEL_DIRS["diffusion_models"],
        "text_encoders":    MODEL_DIRS["text_encoders"],
        "vae":              MODEL_DIRS["vae"],
        "clip_vision":      MODEL_DIRS["clip_vision"],
        # Some nodes also check these aliases:
        "checkpoints":      MODEL_DIRS["diffusion_models"],
        "unet":             MODEL_DIRS["diffusion_models"],
        "clip":             MODEL_DIRS["text_encoders"],
    }

    def get_filename_list(folder_name: str) -> list[str]:
        base = _FOLDER_MAP.get(folder_name)
        if base is None or not base.exists():
            return []
        return [p.name for p in base.iterdir() if p.is_file()]

    def get_full_path(folder_name: str, filename: str) -> str | None:
        base = _FOLDER_MAP.get(folder_name)
        if base is None:
            return None
        candidate = base / filename
        return str(candidate) if candidate.exists() else None

    # Also expose folder_names_and_paths for any code that introspects it.
    shim.folder_names_and_paths = {
        k: ([str(v)], {".safetensors", ".bin", ".pt", ".ckpt"})
        for k, v in _FOLDER_MAP.items()
    }
    shim.get_filename_list = get_filename_list
    shim.get_full_path = get_full_path
    shim.base_path = "/app/framepack"

    sys.modules["folder_paths"] = shim
    print("[patch] folder_paths shim installed.", flush=True)


# =============================================================================
# Pipeline loading (lazy, called once on first handler invocation)
# =============================================================================

def _load_pipeline(job: dict) -> None:
    """
    Import FramePackWrapper nodes and load all four model components into GPU.

    Loading order matters: transformer last (largest allocation).

    IMPORTANT — FramePackWrapper API surface:
    -----------------------------------------
    nodes_framepack.py exposes ComfyUI Node classes.  Each has an execute()
    classmethod whose signature we inspect at runtime to call correctly.
    The key nodes and their return tuples are:

      LoadFramePackModel.execute(model=<filename>)
        -> ({"model": transformer, "scheduler": scheduler},)

      DualCLIPLoader.execute(clip_name1=<llava_file>, clip_name2=<clip_l_file>,
                              type="hunyuan_video")
        -> (clip_obj,)

      VAELoader.execute(vae_name=<vae_file>)
        -> (vae_obj,)

      CLIPVisionLoader.execute(clip_name=<sigclip_file>)
        -> (clip_vision_obj,)

    If any of these APIs change in a future Kijai commit, the ImportError or
    AttributeError will propagate, and the handler returns an error dict with
    a full traceback for diagnosis.
    """
    global _PIPELINE, _PIPELINE_LOADED

    _patch_folder_paths()

    # Add /app/framepack subdirs to sys.path so that relative imports inside
    # the wrapper (e.g. from .utils import ...) resolve correctly.
    for subpath in ["/app/framepack", "/app/framepack/framepack"]:
        if subpath not in sys.path:
            sys.path.insert(0, subpath)

    runpod.serverless.progress_update(job, {"stage": "loading_models"})
    print("[pipeline] Importing FramePackWrapper nodes ...", flush=True)

    # -- Import the node module.  This triggers all top-level torch imports
    #    inside FramePackWrapper (safe; no GPU allocation at import time).
    try:
        from nodes_framepack import (  # type: ignore[import]
            CLIPVisionLoader,
            DualCLIPLoader,
            LoadFramePackModel,
            VAELoader,
        )
    except ImportError as e:
        raise ImportError(
            f"Could not import nodes_framepack from /app/framepack. "
            f"Check that the git clone succeeded and PYTHONPATH is set. "
            f"Original error: {e}"
        ) from e

    # -- VAE (smallest — load first to catch OOM early on small GPUs)
    print("[pipeline] Loading VAE ...", flush=True)
    vae_loader = VAELoader()
    (_PIPELINE["vae"],) = vae_loader.execute(vae_name="hunyuan_video_vae_bf16.safetensors")

    # -- Dual CLIP text encoders (llava_llama3 + clip_l)
    print("[pipeline] Loading dual CLIP text encoders ...", flush=True)
    clip_loader = DualCLIPLoader()
    (_PIPELINE["clip"],) = clip_loader.execute(
        clip_name1="llava_llama3_fp8_scaled.safetensors",
        clip_name2="clip_l.safetensors",
        type="hunyuan_video",
    )

    # -- SigCLIP vision encoder (for I2V image conditioning)
    print("[pipeline] Loading SigCLIP vision encoder ...", flush=True)
    vision_loader = CLIPVisionLoader()
    (_PIPELINE["clip_vision"],) = vision_loader.execute(
        clip_name="sigclip_vision_patch14_384.safetensors"
    )

    # -- FramePack transformer (largest; fp8, ~15 GB — load last)
    print("[pipeline] Loading FramePack transformer ...", flush=True)
    model_loader = LoadFramePackModel()
    (_PIPELINE["framepack_model"],) = model_loader.execute(
        model="FramePackI2V_HY_fp8_e4m3fn.safetensors"
    )

    _PIPELINE_LOADED = True
    print("[pipeline] All models loaded.", flush=True)


# =============================================================================
# Image helpers
# =============================================================================

def _b64_to_pil(image_b64: str) -> Image.Image:
    """Decode a base64-encoded image (any format PIL supports) to RGB PIL."""
    data = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _frames_to_mp4_b64(frames: list[np.ndarray], fps: float = 16.0) -> str:
    """
    Encode a list of uint8 RGB numpy arrays to an MP4 byte string using
    imageio's ffmpeg backend, then base64-encode the result.

    frames: list of H×W×3 uint8 arrays in RGB order.
    fps: output frame rate (FramePack default is 16 fps for I2V).
    """
    import imageio.v3 as iio

    buf = io.BytesIO()
    # imageio-ffmpeg accepts a list of numpy arrays.
    # 'mp4v' codec is universally supported; 'libx264' produces smaller files
    # but requires a full ffmpeg build with x264 support (ubuntu apt ffmpeg has it).
    iio.imwrite(
        buf,
        frames,
        plugin="pyav",
        format="mp4",
        fps=fps,
        codec="libx264",
        output_params=["-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p"],
    )
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# =============================================================================
# Inference — calls FramePackWrapper's FramePackSampler node directly
# =============================================================================

def _generate(job: dict, inp: dict) -> dict:
    """
    Run FramePack I2V inference and return the result dict.

    This calls FramePackWrapper's FramePackSampler ComfyUI node directly via
    its execute() method, bypassing the HTTP ComfyUI server entirely.

    FramePackSampler.execute() signature (as of Kijai 2025-04):
        execute(
            model,            # from LoadFramePackModel
            clip,             # from DualCLIPLoader
            vae,              # from VAELoader
            clip_vision,      # from CLIPVisionLoader
            image,            # torch.Tensor [1, H, W, 3] float32 0-1
            positive,         # text conditioning (CLIP encode result)
            negative,         # negative conditioning
            width, height,    # output resolution
            length,           # number of frames (int)
            steps,            # diffusion steps
            cfg,              # classifier-free guidance scale
            seed,             # RNG seed
            **kwargs          # latent_window_size, gpu_memory_preservation, ...
        ) -> (frames_tensor,)   # [N, H, W, 3] float32 0-1

    CLIPTextEncode-style nodes:
        CLIPTextEncode.execute(text=str, clip=clip_obj) -> (conditioning,)

    Note: if FramePackWrapper's API has drifted from the above (Kijai updates
    frequently), a TypeError/AttributeError will propagate here and be caught
    by the outer try/except in handler(), returning a descriptive error dict.
    """
    import torch
    from nodes_framepack import FramePackSampler  # type: ignore[import]

    # Import CLIPTextEncode from ComfyUI nodes_basic if available, else fall
    # back to a simple wrapper around the clip object's encode method.
    try:
        from nodes import CLIPTextEncode  # type: ignore[import]  # ComfyUI core
    except ImportError:
        # Minimal shim: many ComfyUI CLIP objects expose .encode(text) directly.
        class CLIPTextEncode:  # type: ignore[no-redef]
            @staticmethod
            def execute(text: str, clip: Any) -> tuple:
                conditioning = clip.encode(text)
                return (conditioning,)

    # -- Decode and convert input image to torch tensor [1, H, W, 3] float32
    pil_img = _b64_to_pil(inp["image_b64"])
    width:  int   = int(inp.get("width",  768))
    height: int   = int(inp.get("height", 432))
    pil_img = pil_img.resize((width, height), Image.LANCZOS)
    img_np  = np.array(pil_img, dtype=np.float32) / 255.0   # H×W×3
    img_t   = torch.from_numpy(img_np).unsqueeze(0)          # 1×H×W×3

    duration_s: float = float(inp.get("duration_s", 10))
    fps:        float = 16.0         # FramePack I2V canonical output rate
    n_frames:   int   = max(1, int(duration_s * fps))
    steps:      int   = int(inp.get("steps", 25))
    cfg:        float = float(inp.get("cfg", 7.0))
    seed:       int   = int(inp.get("seed", 42))

    prompt:          str = inp["prompt"]
    negative_prompt: str = inp.get("negative_prompt", "blurry, low quality, distorted")

    # -- Encode text prompts
    clip_encoder = CLIPTextEncode()
    (positive,) = clip_encoder.execute(text=prompt,          clip=_PIPELINE["clip"])
    (negative,) = clip_encoder.execute(text=negative_prompt, clip=_PIPELINE["clip"])

    runpod.serverless.progress_update(job, {
        "stage": "sampling",
        "steps": steps,
        "frames": n_frames,
    })

    # -- Run sampler
    sampler = FramePackSampler()
    (frames_tensor,) = sampler.execute(
        model       = _PIPELINE["framepack_model"],
        clip        = _PIPELINE["clip"],
        vae         = _PIPELINE["vae"],
        clip_vision = _PIPELINE["clip_vision"],
        image       = img_t,
        positive    = positive,
        negative    = negative,
        width       = width,
        height      = height,
        length      = n_frames,
        steps       = steps,
        cfg         = cfg,
        seed        = seed,
    )

    # frames_tensor shape: [N, H, W, 3] float32 0-1  (FramePack convention)
    frames_np: list[np.ndarray] = [
        (frames_tensor[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        for i in range(frames_tensor.shape[0])
    ]

    runpod.serverless.progress_update(job, {"stage": "encoding_video"})
    video_b64 = _frames_to_mp4_b64(frames_np, fps=fps)

    return {
        "video_b64":  video_b64,
        "duration_s": len(frames_np) / fps,
        "frames":     len(frames_np),
    }


# ---------------------------------------------------------------------------
# FALLBACK — headless ComfyUI subprocess approach
# ---------------------------------------------------------------------------
# If the direct Python API above breaks due to tight ComfyUI coupling, you
# can replace _generate() with a subprocess approach:
#
#   1. Install ComfyUI into /app/comfyui in the Dockerfile.
#   2. Write a workflow JSON to disk.
#   3. Launch: subprocess.Popen(["python", "/app/comfyui/main.py",
#                                "--listen", "127.0.0.1", "--port", "8188",
#                                "--extra-model-paths-config", <yaml>])
#   4. Wait for the HTTP server to be ready (poll /health).
#   5. POST the workflow to /prompt and poll /history/<prompt_id>.
#   6. Fetch the output video bytes from /view.
#
# This adds ~10 s to cold start and ~400 MB to the image but is completely
# decoupled from the Python import structure.  Uncomment and adapt as needed.
# ---------------------------------------------------------------------------


# =============================================================================
# RunPod Serverless entry point
# =============================================================================

def handler(job: dict) -> dict:
    """
    RunPod Serverless handler.

    Input schema (job["input"]):
        image_b64        (str, required)  — base64-encoded source image (any common format)
        prompt           (str, required)  — text description of desired motion/scene
        negative_prompt  (str, optional)  — default: "blurry, low quality, distorted"
        duration_s       (float, optional)— clip length in seconds; default 10
        steps            (int, optional)  — diffusion steps; default 25
        cfg              (float, optional)— CFG guidance scale; default 7.0
        seed             (int, optional)  — RNG seed for reproducibility; default 42
        width            (int, optional)  — output width px; default 768
        height           (int, optional)  — output height px; default 432

    Output:
        video_b64  (str)   — base64-encoded MP4 video
        duration_s (float) — actual video duration in seconds
        frames     (int)   — number of frames generated
        elapsed_s  (float) — wall-clock seconds for this invocation

    On error:
        error      (str)   — exception message
        traceback  (str)   — full Python traceback for diagnosis
    """
    t0 = time.time()

    try:
        inp = job.get("input", {})

        # -- Validate required fields
        if "image_b64" not in inp:
            return {"error": "Missing required input field: image_b64"}
        if "prompt" not in inp:
            return {"error": "Missing required input field: prompt"}

        # -- Ensure models are on disk (idempotent; skips if already present)
        runpod.serverless.progress_update(job, {"stage": "checking_models"})
        ensure_models_downloaded(job)

        # -- Load pipeline once; reuse on subsequent calls
        global _PIPELINE_LOADED
        if not _PIPELINE_LOADED:
            _load_pipeline(job)

        # -- Run inference
        result = _generate(job, inp)
        result["elapsed_s"] = round(time.time() - t0, 2)
        return result

    except Exception as exc:  # noqa: BLE001
        return {
            "error":     str(exc),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Start the RunPod serverless worker loop when executed directly.
# RunPod calls handler() once per queued job; the loop is managed by the SDK.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
