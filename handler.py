"""
handler.py — FramePack I2V RunPod Serverless Worker (ComfyUI subprocess pattern)
==================================================================================

Architecture
------------
ComfyUI runs as a child process inside the container on 127.0.0.1:8188.
This handler talks to it over localhost HTTP using the standard ComfyUI API:
  POST /prompt      — queue a workflow
  GET  /history/:id — poll for completion
  POST /upload/image — upload the keyframe PNG
  GET  /view        — fetch the finished MP4

This avoids the ModuleNotFoundError that occurs when attempting to import
ComfyUI custom nodes (e.g. nodes_framepack) as bare Python modules — custom
nodes are designed to run inside the ComfyUI node system, not standalone.

RunPod contract
---------------
RunPod calls handler(job) once per request. job["input"] carries the user
payload. The function must return a plain dict. On error, return
{"error": str, "traceback": str}.

Global state
------------
_COMFY_PROC — the Popen handle; None until start_comfyui() succeeds.
_COMFY_READY — True once /system_stats returns 200 and nodes are verified.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import subprocess
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import runpod

# ---------------------------------------------------------------------------
# ComfyUI subprocess state
# ---------------------------------------------------------------------------
_COMFY_PROC: subprocess.Popen | None = None
_COMFY_READY: bool = False
_BASE_URL = "http://127.0.0.1:8188"

# ---------------------------------------------------------------------------
# Model manifest (verbatim from remote_models_framepack.yml)
# ---------------------------------------------------------------------------
VOLUME_ROOT = Path("/runpod-volume/models")

MODEL_MANIFEST = [
    {
        "subdir":     "diffusion_models",
        "filename":   "FramePackI2V_HY_fp8_e4m3fn.safetensors",
        "url":        "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/FramePackI2V_HY_fp8_e4m3fn.safetensors",
        "size_bytes": 16_331_849_976,
    },
    {
        "subdir":     "text_encoders",
        "filename":   "llava_llama3_fp8_scaled.safetensors",
        "url":        "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors",
        "size_bytes": 9_091_392_483,
    },
    {
        "subdir":     "text_encoders",
        "filename":   "clip_l.safetensors",
        "url":        "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/clip_l.safetensors",
        "size_bytes": 246_144_152,
    },
    {
        "subdir":     "vae",
        "filename":   "hunyuan_video_vae_bf16.safetensors",
        "url":        "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/hunyuan_video_vae_bf16.safetensors",
        "size_bytes": 492_986_478,
    },
    {
        "subdir":     "clip_vision",
        "filename":   "sigclip_vision_patch14_384.safetensors",
        "url":        "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors",
        "size_bytes": 856_505_640,
    },
]

# Required ComfyUI node classes — verified at startup
REQUIRED_NODES = ["FramePackSampler", "LoadFramePackModel", "DualCLIPLoader"]


# =============================================================================
# Model download helpers
# =============================================================================

def _needs_download(entry: dict) -> bool:
    dest = VOLUME_ROOT / entry["subdir"] / entry["filename"]
    if not dest.exists():
        return True
    # Re-download if file is smaller than 5% of expected size
    return dest.stat().st_size < (entry["size_bytes"] * 0.05)


def _download_one(entry: dict) -> str:
    dest_dir = VOLUME_ROOT / entry["subdir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / entry["filename"]

    hf_token = os.environ.get("HF_TOKEN", "")
    auth_header = ["-H", f"Authorization: Bearer {hf_token}"] if hf_token else []

    cmd = [
        "curl", "-L", "--fail", "-C", "-",
        *auth_header,
        "-o", str(dest),
        entry["url"],
    ]
    print(f"[download] Starting: {entry['filename']}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"curl failed for {entry['filename']} (rc={result.returncode}): {result.stderr[:500]}"
        )
    print(f"[download] Done: {entry['filename']}", flush=True)
    return entry["filename"]


def ensure_models(job: dict) -> None:
    """Download any missing or incomplete model files to the network volume."""
    to_download = [e for e in MODEL_MANIFEST if _needs_download(e)]
    if not to_download:
        print("[models] All model files verified on volume.", flush=True)
        return

    total_gb = sum(e["size_bytes"] for e in to_download) / 1e9
    print(f"[models] {len(to_download)} file(s) need downloading ({total_gb:.1f} GB total).", flush=True)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_download_one, e): e for e in to_download}
        for future in as_completed(futures):
            entry = futures[future]
            runpod.serverless.progress_update(job, {"stage": f"downloading {entry['filename']}"})
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed to download {entry['filename']}: {exc}") from exc


# =============================================================================
# ComfyUI subprocess management
# =============================================================================

def start_comfyui() -> None:
    """Launch ComfyUI as a subprocess and wait until it is ready to serve requests."""
    global _COMFY_PROC, _COMFY_READY

    if _COMFY_READY and _COMFY_PROC is not None and _COMFY_PROC.poll() is None:
        return

    print("[comfyui] Starting ComfyUI subprocess...", flush=True)
    log_file = open("/tmp/comfy.log", "a")
    _COMFY_PROC = subprocess.Popen(
        ["python", "/app/ComfyUI/main.py", "--listen", "127.0.0.1", "--port", "8188"],
        stdout=log_file,
        stderr=log_file,
    )

    # Poll /system_stats until ready (up to 180 s)
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            resp = requests.get(f"{_BASE_URL}/system_stats", timeout=3)
            if resp.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    else:
        raise RuntimeError("ComfyUI did not become ready within 180 seconds. Check /tmp/comfy.log.")

    print("[comfyui] ComfyUI is up. Verifying required nodes...", flush=True)

    # Verify all required custom nodes are registered
    resp = requests.get(f"{_BASE_URL}/object_info", timeout=30)
    resp.raise_for_status()
    available = set(resp.json().keys())
    missing = [n for n in REQUIRED_NODES if n not in available]
    if missing:
        raise RuntimeError(
            f"ComfyUI is missing required nodes: {missing}. "
            f"Check that ComfyUI-FramePackWrapper was cloned correctly into custom_nodes."
        )

    print(f"[comfyui] All required nodes present: {REQUIRED_NODES}", flush=True)
    _COMFY_READY = True


# =============================================================================
# ComfyUI API helpers
# =============================================================================

def upload_image(local_path: str, remote_name: str) -> str:
    """Upload a PNG to ComfyUI's /upload/image endpoint. Returns the saved filename."""
    with open(local_path, "rb") as f:
        resp = requests.post(
            f"{_BASE_URL}/upload/image",
            files={"image": (remote_name, f, "image/png")},
            timeout=30,
        )
    resp.raise_for_status()
    data = resp.json()
    return data.get("name", remote_name)


def queue_workflow(workflow: dict) -> str:
    """POST a workflow to /prompt and return the prompt_id."""
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    resp = requests.post(f"{_BASE_URL}/prompt", json=payload, timeout=30)
    if resp.status_code == 400:
        print(f"[comfyui] /prompt 400 error: {resp.text[:1500]}", flush=True)
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_prompt(prompt_id: str, timeout: int = 1800) -> dict:
    """Poll /history/:id every 2 s until outputs appear. Returns the history entry."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{_BASE_URL}/history/{prompt_id}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if prompt_id in data:
            entry = data[prompt_id]
            if entry.get("outputs"):
                return entry
        time.sleep(2)
    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout} seconds.")


def fetch_output_video(entry: dict) -> bytes:
    """Walk history entry outputs, find the first .mp4, and download it."""
    for _node_id, node_outputs in entry.get("outputs", {}).items():
        for key, items in node_outputs.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and item.get("filename", "").endswith(".mp4"):
                    params = {
                        "filename": item["filename"],
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                    }
                    resp = requests.get(f"{_BASE_URL}/view", params=params, timeout=120)
                    resp.raise_for_status()
                    return resp.content
    raise RuntimeError(
        f"No .mp4 found in workflow outputs. Available outputs: {list(entry.get('outputs', {}).keys())}"
    )


# =============================================================================
# Workflow patching
# =============================================================================

def _patch_workflow(workflow: dict, inp: dict, uploaded_image_name: str) -> dict:
    """
    Deep-copy workflow and patch user-supplied values by _meta.title.

    Patched nodes:
      'Positive Prompt'   -> inputs.text  = prompt
      'Negative Prompt'   -> inputs.text  = negative_prompt
      'FramePack Sampler' -> inputs.total_second_length, steps, cfg, seed
      'Input Keyframe'    -> inputs.image = uploaded_image_name
    """
    wf = copy.deepcopy(workflow)

    prompt = inp["prompt"]
    negative_prompt = inp.get("negative_prompt", "")
    duration_s = float(inp.get("duration_s", 10))
    steps = int(inp.get("steps", 25))
    cfg = float(inp.get("cfg", 7.0))
    seed = int(inp.get("seed", 42))

    for node in wf.values():
        if not isinstance(node, dict):
            continue
        title = node.get("_meta", {}).get("title", "")
        inputs = node.get("inputs", {})

        if title == "Positive Prompt":
            inputs["text"] = prompt
        elif title == "Negative Prompt":
            inputs["text"] = negative_prompt
        elif title == "FramePack Sampler":
            inputs["total_second_length"] = duration_s
            inputs["steps"] = steps
            inputs["cfg"] = cfg
            inputs["seed"] = seed
        elif title == "Input Keyframe":
            inputs["image"] = uploaded_image_name

    return wf


# =============================================================================
# RunPod Serverless entry point
# =============================================================================

def handler(job: dict) -> dict:
    """
    RunPod Serverless handler.

    Input schema (job["input"]):
        image_b64        (str, required)  — base64-encoded source image
        prompt           (str, required)  — text description of desired motion/scene
        negative_prompt  (str, optional)  — default: ""
        duration_s       (float, optional)— clip length in seconds; default 10
        steps            (int, optional)  — diffusion steps; default 25
        cfg              (float, optional)— CFG guidance scale; default 7.0
        seed             (int, optional)  — RNG seed; default 42

    Output:
        video_b64  (str)   — base64-encoded MP4
        duration_s (float) — requested duration
        elapsed_s  (float) — wall-clock seconds for this job

    On error:
        error      (str)   — exception message
        traceback  (str)   — full traceback
    """
    t0 = time.time()
    log = open("/tmp/comfy.log", "a")

    try:
        inp = job.get("input", {})

        if "image_b64" not in inp:
            return {"error": "Missing required input field: image_b64"}
        if "prompt" not in inp:
            return {"error": "Missing required input field: prompt"}

        # 1. Decode input image
        runpod.serverless.progress_update(job, {"stage": "decoding image"})
        img_bytes = base64.b64decode(inp["image_b64"])
        tmp_img = "/tmp/kf.png"
        with open(tmp_img, "wb") as f:
            f.write(img_bytes)
        print(f"[handler] Input image written to {tmp_img}", flush=True)

        # 2. Ensure models are on disk
        runpod.serverless.progress_update(job, {"stage": "checking models"})
        ensure_models(job)

        # 3. Start ComfyUI (no-op if already running)
        runpod.serverless.progress_update(job, {"stage": "starting ComfyUI"})
        start_comfyui()

        # 4. Upload keyframe to ComfyUI
        runpod.serverless.progress_update(job, {"stage": "uploading keyframe"})
        remote_name = upload_image(tmp_img, "input_keyframe.png")
        print(f"[handler] Keyframe uploaded as: {remote_name}", flush=True)

        # 5. Load and patch workflow
        runpod.serverless.progress_update(job, {"stage": "queuing workflow"})
        with open("/app/workflow.json") as f:
            base_workflow = json.load(f)
        patched = _patch_workflow(base_workflow, inp, remote_name)

        # 6. Queue workflow and wait for result
        prompt_id = queue_workflow(patched)
        print(f"[handler] Queued prompt_id: {prompt_id}", flush=True)

        runpod.serverless.progress_update(job, {"stage": "sampling", "prompt_id": prompt_id})
        entry = wait_for_prompt(prompt_id, timeout=1800)

        # 7. Fetch and encode output video
        runpod.serverless.progress_update(job, {"stage": "encoding video"})
        video_bytes = fetch_output_video(entry)
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")

        elapsed = round(time.time() - t0, 2)
        print(f"[handler] Done in {elapsed}s. Video size: {len(video_bytes) / 1e6:.1f} MB", flush=True)

        return {
            "video_b64": video_b64,
            "duration_s": float(inp.get("duration_s", 10)),
            "elapsed_s": elapsed,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        msg = f"[handler] ERROR: {exc}\n{tb}"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        return {"error": str(exc), "traceback": tb}

    finally:
        log.close()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
