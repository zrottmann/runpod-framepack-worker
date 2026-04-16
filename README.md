# FramePack I2V — RunPod Serverless Worker

Generates short anime-style videos from a single image using FramePack I2V
(HunyuanVideo-based diffusion transformer, fp8 quantised, ~12 GB VRAM).

---

## How it works

The worker uses the **ComfyUI subprocess pattern**:

1. `entrypoint.sh` symlinks the RunPod network volume model dirs into
   `/app/ComfyUI/models/` and then execs `handler.py`.
2. On first job, `handler.py` downloads any missing model weights from
   HuggingFace in parallel (5 threads, `curl -L --fail -C -`).
3. `handler.py` launches ComfyUI (`/app/ComfyUI/main.py`) as a background
   subprocess on `127.0.0.1:8188`, then polls `/system_stats` until ready
   and verifies `FramePackSampler`, `LoadFramePackModel`, and `DualCLIPLoader`
   are registered via `/object_info`.
4. The keyframe image is uploaded to ComfyUI via `POST /upload/image`.
5. The bundled `workflow.json` (copied from
   `workflows/framepack_long_i2v_remote.json` at build time) is deep-copied
   and patched with the job's `prompt`, `negative_prompt`, `duration_s`,
   `steps`, `cfg`, and `seed` values by `_meta.title` matching.
6. The patched workflow is queued via `POST /prompt`; completion is polled
   at `GET /history/:id`.
7. The finished `.mp4` is downloaded from `GET /view` and returned as
   base64 in `video_b64`.

This pattern is completely decoupled from ComfyUI's Python internals — no
direct imports of custom node modules are needed.

---

## Architecture

| Layer | What lives there |
|---|---|
| Docker image (~3–4 GB) | Python, PyTorch, ComfyUI, FramePackWrapper, VideoHelperSuite |
| RunPod Network Volume (50 GB) | All model weights (~25.2 GB total) |

Custom nodes bundled in the image:
- `ComfyUI-FramePackWrapper` (kijai/ComfyUI-FramePackWrapper)
- `ComfyUI-VideoHelperSuite` (Kosinkadink/ComfyUI-VideoHelperSuite)

On first invocation the handler downloads 5 weight files from HuggingFace
in parallel. Every subsequent invocation skips the download (file-size check).
Cold start after download: ~30–60 s. Warm (ComfyUI already up): ~2 s overhead.

---

## Step 1 — Push to GitHub

```bash
cd /path/to/dorohedoro_anime_maker/runpod_endpoint

git init
git add .
git commit -m "Initial FramePack I2V serverless worker"
git remote add origin https://github.com/YOUR_USERNAME/runpod-framepack-worker.git
git branch -M main
git push -u origin main
```

---

## Step 2 — Create Network Volume

In the RunPod Console:

1. **Storage → Network Volumes → + New Network Volume**
2. Name: `framepack-models`  |  Size: **50 GB**  |  Region: near your GPU datacenter.
3. Note the volume ID for Step 3.

---

## Step 3 — Create the Serverless Endpoint

1. **Serverless → + New Endpoint**
2. Select **Custom Source → GitHub Repo**
3. Paste: `https://github.com/YOUR_USERNAME/runpod-framepack-worker`
4. Branch: `main`
5. **GPU:** RTX 4090 (24 GB VRAM) recommended.
6. **Min/Max Workers:** 0 / 1 for low-volume use.
7. **Network Volume:** attach `framepack-models` at `/runpod-volume`
8. **Environment Variables:**
   - `HF_TOKEN` — HuggingFace read token (recommended to avoid rate limits)
9. Click **Deploy**. Build takes 5–10 min (~3–4 GB image).

---

## Step 4 — Call the Endpoint

### Python (runpod SDK)

```python
import runpod, base64, pathlib

runpod.api_key = "YOUR_RUNPOD_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

image_b64 = base64.b64encode(pathlib.Path("my_image.png").read_bytes()).decode()

result = endpoint.run_sync({
    "input": {
        "image_b64":       image_b64,
        "prompt":          "A ninja leaping across rooftops at night, cinematic",
        "negative_prompt": "blurry, low quality, static, no movement",
        "duration_s":      6,
        "steps":           25,
        "cfg":             7.0,
        "seed":            42,
    }
}, timeout=600)

if "error" in result:
    print("Error:", result["error"])
    print(result.get("traceback", ""))
else:
    video_bytes = base64.b64decode(result["video_b64"])
    pathlib.Path("output.mp4").write_bytes(video_bytes)
    print(f"Saved output.mp4  ({result['duration_s']}s, {result['elapsed_s']}s elapsed)")
```

### cURL

```bash
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
```

---

## Input Schema

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_b64` | string | Yes | — | Base64-encoded source image (JPEG/PNG) |
| `prompt` | string | Yes | — | Text description of motion and scene |
| `negative_prompt` | string | No | `""` | What to avoid |
| `duration_s` | float | No | `10` | Video length in seconds |
| `steps` | int | No | `25` | Diffusion sampling steps |
| `cfg` | float | No | `7.0` | CFG guidance scale |
| `seed` | int | No | `42` | Random seed |

## Output Schema

| Field | Type | Description |
|---|---|---|
| `video_b64` | string | Base64-encoded MP4 |
| `duration_s` | float | Requested duration |
| `elapsed_s` | float | Wall-clock time for this job |
| `error` | string | Present only on failure |
| `traceback` | string | Present only on failure |

---

## Troubleshooting

**"ComfyUI is missing required nodes"** — The custom node git clone in the
Dockerfile failed. Check the build log for clone errors.

**ComfyUI not ready within 180 s** — Check `/tmp/comfy.log` in the container
logs for import errors. Usually a missing Python dependency.

**OOM on first invocation** — The fp8 transformer needs ~12 GB VRAM. Use
RTX 4090 (24 GB) or A100.

**Download stuck** — Set `HF_TOKEN` in the endpoint environment variables
to avoid HuggingFace anonymous rate limits.
