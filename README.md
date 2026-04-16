# FramePack I2V — RunPod Serverless Worker

Generates short anime-style videos from a single image using FramePack I2V
(HunyuanVideo-based diffusion transformer, fp8 quantised, ~12 GB VRAM).

---

## Architecture

| Layer | What lives there |
|---|---|
| Docker image (~2 GB) | Python code, PyTorch, FramePackWrapper |
| RunPod Network Volume (50 GB) | All model weights (~25.2 GB total) |

On first invocation the handler downloads the 5 weight files from
HuggingFace in parallel.  Every subsequent invocation skips the download.
Cold start after download: ~30–60 s.  Warm (already loaded in GPU): ~2 s.

---

## Step 1 — Push to GitHub

```bash
# From inside runpod_endpoint/ (this directory):
cd /path/to/dorohedoro_anime_maker/runpod_endpoint

git init
git add .
git commit -m "Initial FramePack I2V serverless worker"

# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/runpod-framepack-worker.git
git branch -M main
git push -u origin main
```

**Private repo:** You must add a Deploy Key or Personal Access Token (PAT) in
the RunPod Console under Settings → Secrets → GitHub so RunPod can pull the
repo.  A public repo requires no token.

---

## Step 2 — Create Network Volume

In the RunPod Console:

1. **Storage → Network Volumes → + New Network Volume**
2. Name: `framepack-models`  |  Size: **50 GB**  |  Region: pick one near
   your preferred GPU datacenter.
3. Note the volume ID for Step 3.

---

## Step 3 — Create the Serverless Endpoint

1. **Serverless → + New Endpoint**
2. Select **Custom Source → GitHub Repo**
3. Paste your repo URL: `https://github.com/YOUR_USERNAME/runpod-framepack-worker`
4. Branch: `main`
5. **GPU:** RTX 4090 (24 GB VRAM) recommended — fp8 transformer fits in 12 GB
   VRAM, leaving headroom for VAE decode and CLIP.
   - On-demand spot price: ~$0.34/hr (as of 2025)
   - Idle workers: ~$0.09/hr on spot
6. **Min/Max Workers:** 0 / 1 for low-volume use; increase Max for parallelism.
7. **Network Volume:** attach `framepack-models` at mount path `/runpod-volume`
8. **Environment Variables** (optional but recommended):
   - `HF_TOKEN` — your HuggingFace read token (all models here are public, so
     only needed if you hit anonymous rate limits)
9. Click **Deploy**.

Build takes 5–8 minutes.  Watch the build log; the image is ~2 GB.

---

## Step 4 — Call the Endpoint

### Python (runpod SDK)

```python
import runpod, base64, pathlib

runpod.api_key = "YOUR_RUNPOD_API_KEY"  # from RunPod Console → Settings → API Keys
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")  # from Serverless → your endpoint

# Read and encode your source image
image_b64 = base64.b64encode(pathlib.Path("my_image.png").read_bytes()).decode()

result = endpoint.run_sync({
    "input": {
        "image_b64":       image_b64,
        "prompt":          "A ninja leaping across rooftops at night, cinematic",
        "negative_prompt": "blurry, low quality, static, no movement",
        "duration_s":      6,       # seconds of video
        "steps":           25,
        "cfg":             7.0,
        "seed":            42,
        "width":           768,
        "height":          432,
    }
}, timeout=600)  # first-run can take 10+ min for model download

if "error" in result:
    print("Error:", result["error"])
    print(result.get("traceback", ""))
else:
    video_bytes = base64.b64decode(result["video_b64"])
    pathlib.Path("output.mp4").write_bytes(video_bytes)
    print(f"Saved output.mp4  ({result['frames']} frames, {result['duration_s']}s, {result['elapsed_s']}s elapsed)")
```

### cURL

```bash
# Submit a job
curl -s -X POST "https://api.runpod.io/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_b64": "'$(base64 -w0 my_image.png)'",
      "prompt": "A ninja leaping across rooftops at night",
      "duration_s": 6,
      "steps": 25,
      "cfg": 7.0,
      "seed": 42
    }
  }'
# Returns: {"id": "JOB_ID", "status": "IN_QUEUE"}

# Poll for result
curl -s "https://api.runpod.io/v2/YOUR_ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"
```

---

## Input Schema

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_b64` | string | Yes | — | Base64-encoded source image (JPEG/PNG/etc.) |
| `prompt` | string | Yes | — | Text description of motion and scene |
| `negative_prompt` | string | No | `"blurry, low quality, distorted"` | What to avoid |
| `duration_s` | float | No | `10` | Video length in seconds |
| `steps` | int | No | `25` | Diffusion sampling steps (more = slower + better) |
| `cfg` | float | No | `7.0` | CFG guidance scale |
| `seed` | int | No | `42` | Random seed for reproducibility |
| `width` | int | No | `768` | Output width in pixels |
| `height` | int | No | `432` | Output height in pixels |

## Output Schema

| Field | Type | Description |
|---|---|---|
| `video_b64` | string | Base64-encoded MP4 |
| `duration_s` | float | Actual video duration |
| `frames` | int | Frame count |
| `elapsed_s` | float | Wall-clock time for this job |
| `error` | string | Present only on failure |
| `traceback` | string | Present only on failure |

---

## Troubleshooting

**"Could not import nodes_framepack"** — The FramePackWrapper git clone in the
Dockerfile failed, or `PYTHONPATH` is not set.  Check the build log.

**OOM on first invocation** — The fp8 transformer needs ~12 GB VRAM.  Use an
RTX 4090 (24 GB) or A100.  An RTX 3090 (24 GB) also works but is slower.

**Download stuck** — HuggingFace occasionally rate-limits anonymous requests.
Set the `HF_TOKEN` environment variable in the endpoint config.

**`libx264` not found** — Ubuntu 22.04's `apt` ffmpeg includes libx264.  If you
see a codec error, the imageio encoder falls back to `mp4v` automatically;
output quality is slightly lower but otherwise identical.
