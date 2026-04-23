# Deploy `Final_ecg_project` on Render

This folder is now ready to run as a standalone ECG inference API.

## 1) Push as a separate Git repo
Render deploys from a repository root. Put this folder in its own repo, or make this folder the repo root.

## 2) Create Render Web Service
- Render Dashboard -> New -> Web Service
- Connect your repo
- Runtime: `Docker`
- Dockerfile path: `./Dockerfile`

You can also use `render.yaml` in this folder for blueprint-based setup.

## 3) Environment variable
Set:
- `ECG_MODEL_PATH=/app/model/resnet_ecg.h5`

## 4) Health check
After deploy:

```bash
curl https://<your-service>.onrender.com/health
```

Expected shape:

```json
{"ok": true, "modelLoaded": true, "classes": ["HB","MI","PMI","Normal"]}
```

## 5) Predict endpoint (multipart file upload)

```bash
curl -X POST "https://<your-service>.onrender.com/predict" \
  -F "image=@/absolute/path/to/ecg.jpg"
```

## 6) Predict endpoint (base64)

```bash
curl -X POST "https://<your-service>.onrender.com/predict-base64" \
  -H "Content-Type: application/json" \
  -d '{"imageBase64":"data:image/jpeg;base64,<base64_here>"}'
```

## Response format

```json
{
  "label": "MI",
  "confidence": 93.42,
  "probabilities": {
    "HB": 1.05,
    "MI": 93.42,
    "PMI": 2.11,
    "Normal": 3.42
  }
}
```
