# BraTS 3D Tumor Visualization - Deployment Guide

## Architecture Overview

```
┌─────────────────────┐         ┌──────────────────────┐
│     Vercel          │  HTTP   │   RunPod Serverless  │
│   (Next.js UI)      │ ──────▶ │   (GPU Inference)    │
│                     │ ◀────── │                      │
│  - Upload MRI       │         │  - StableUNet3D      │
│  - 3D Visualization │         │  - Mesh Generation   │
│  - Report Display   │         │  - GLTF Export       │
└─────────────────────┘         └──────────────────────┘
```

---

## Part 1: Deploy to RunPod Serverless

### Step 1: Build Docker Image

```bash
cd BraTS_Optimized_Solution

# Build the image
docker build -f Dockerfile.runpod -t YOUR_DOCKERHUB/brats-runpod:latest .

# Test locally (optional)
docker run --gpus all -p 8000:8000 YOUR_DOCKERHUB/brats-runpod:latest

# Push to Docker Hub
docker login
docker push YOUR_DOCKERHUB/brats-runpod:latest
```

### Step 2: Create RunPod Serverless Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click **"New Endpoint"**
3. Configure:
   - **Name**: `brats-inference`
   - **Container Image**: `YOUR_DOCKERHUB/brats-runpod:latest`
   - **GPU Type**: `RTX A4000` (16GB) or `RTX 3090` (24GB)
   - **Min Workers**: `0` (scale to zero)
   - **Max Workers**: `3` (adjust based on usage)
   - **Idle Timeout**: `60` seconds
   - **Flash Boot**: ✅ Enable (faster cold starts)
   
4. Click **Create Endpoint**
5. Copy the **Endpoint ID** (e.g., `abc123xyz`)

### Step 3: Get RunPod API Key

1. Go to [RunPod Settings > API Keys](https://www.runpod.io/console/user/settings)
2. Click **"Create API Key"**
3. Copy the key (starts with `rp_...`)

---

## Part 2: Deploy UI to Vercel

### Step 1: Push to GitHub

```bash
cd ui_design/brats-viewer-ui
git add -A
git commit -m "Add RunPod serverless integration"
git push origin main
```

### Step 2: Import to Vercel

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click **"Add New Project"**
3. Select your GitHub repository
4. Configure:
   - **Framework Preset**: Next.js
   - **Root Directory**: `ui_design/brats-viewer-ui`
   
5. Add Environment Variables:
   | Name | Value |
   |------|-------|
   | `NEXT_PUBLIC_RUNPOD_ENDPOINT_ID` | Your RunPod endpoint ID |
   | `NEXT_PUBLIC_RUNPOD_API_KEY` | Your RunPod API key |

6. Click **Deploy**

---

## Part 3: Testing the Deployment

### Test RunPod Endpoint Directly

```bash
# Health check
curl -X GET https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/health \
  -H "Authorization: Bearer YOUR_API_KEY"

# Submit test job (with sample data)
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "files": [
        {"filename": "t1.nii.gz", "content": "BASE64_ENCODED_FILE"},
        {"filename": "t1ce.nii.gz", "content": "BASE64_ENCODED_FILE"},
        {"filename": "t2.nii.gz", "content": "BASE64_ENCODED_FILE"},
        {"filename": "flair.nii.gz", "content": "BASE64_ENCODED_FILE"}
      ],
      "options": {"generate_report": true}
    }
  }'

# Check job status
curl -X GET https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Test Full Pipeline

1. Open your Vercel deployment URL
2. Upload 4 MRI modality files (T1, T1ce, T2, FLAIR)
3. Wait for processing (should show progress)
4. View 3D tumor visualization
5. Download GLTF for AR viewing

---

## Cost Estimation

### RunPod Serverless (GPU)

| GPU | Price/sec | 1 inference (~60s) | 100 inferences |
|-----|-----------|-------------------|----------------|
| RTX A4000 | $0.00031 | ~$0.02 | ~$2.00 |
| RTX 3090 | $0.00044 | ~$0.03 | ~$3.00 |
| RTX A5000 | $0.00036 | ~$0.02 | ~$2.20 |

### Vercel (Frontend)

- **Hobby Plan**: Free (sufficient for demos)
- **Pro Plan**: $20/month (for production)

---

## Troubleshooting

### Cold Start Latency

RunPod serverless may take 30-60 seconds to start on first request.
- **Solution**: Enable Flash Boot or keep 1 warm worker

### Large File Upload Failures

MRI files can be 50-200MB each.
- **Solution**: Files are base64 encoded; ensure request size < 10MB per file
- For larger files, consider pre-signed URL upload to cloud storage

### CORS Errors

If you see CORS errors in browser:
- RunPod API handles CORS automatically
- Ensure Vercel headers are correctly configured in `vercel.json`

### GPU Out of Memory

If inference fails with OOM:
- Upgrade to larger GPU (A4000 → A5000 → A100)
- Disable TTA by setting `tta_enabled: false` in options

---

## Local Development

```bash
# Frontend
cd ui_design/brats-viewer-ui
cp .env.example .env.local
# Edit .env.local with your RunPod credentials
npm install
npm run dev

# Backend (optional, for testing without RunPod)
cd BraTS_Optimized_Solution
pip install -r requirements_backend.txt
python -m uvicorn api_server:app --reload --port 8000
```

---

## Security Notes

⚠️ **Never commit API keys to git!**

- Use Vercel environment variables for production
- Use `.env.local` for local development (gitignored)
- Rotate API keys periodically
