# OCR Demo with OpenVINO Model Server on OpenShift AI

Text recognition demo using TrOCR model deployed on Red Hat OpenShift AI (RHOAI) with OpenVINO Model Server (OVMS).

## Prerequisites

- Red Hat OpenShift AI (RHOAI)
- S3-compatible object storage (MinIO)

## Setup

### 1. Create Project
Project name: `ocr-demo`

### 2. Configure Object Storage
- **Bucket**: Create bucket named `ocr-demo`
- **Data Connection**:
  - Type: S3 compatible object storage
  - Name: `minio`
  - Access key: `<your-access-key>`
  - Secret key: `<your-secret-key>`
  - Endpoint: `https://<minio-route>`
  - Region: `None`
  - Bucket: `ocr-demo`

### 3. Create Workbench
- **Name**: `workbench`
- **Image**: `Jupyter | Minimal | CPU | Python 3.12`
- **Memory**: 8 GiB (request & limit)
- **Connections**: Attach `minio`
- **Git repo**: Clone this repository

### 4. Prepare Model
Run notebooks **0 through 4** in order. These will:
- Download TrOCR model from Hugging Face
- Convert to ONNX format
- Upload to S3 with required structure (`models/1/model.onnx`)

### 5. Deploy Model
- **Model name**: `onnx-model`
- **Connection**: `minio`
- **Path**: `models`
- **Framework**: `onnx - 1`
- **Runtime**: OpenVINO Model Server (auto-selected)
- **External route**: ✅ Enabled
- **Authentication**: ❌ Disabled

### 6. Run Inference
Execute **notebook 5** to test OCR inference via the deployed OVMS endpoint.

## Notebooks

0. Explore model from Hugging Face
1. Save model to local storage in ONNX
2. Test local ONNX inference
3. Upload to S3
4. Test S3-based inference
5. Test OVMS remote inference
