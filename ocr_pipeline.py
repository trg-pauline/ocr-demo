from kfp import dsl, compiler
from kfp.dsl import Output, Artifact, Input


BASE_IMAGE = "registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:8712328f7c4aa611500daf3736a53405cac8a23a97721c37745a9fb9d8cf321d"
EXTRA_INDEX = "https://pypi.org/simple/"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "optimum[exporters,onnxruntime]==2.1.0",
        "transformers==4.57.6",
        "torch==2.11.0",
    ],
    pip_index_urls=[EXTRA_INDEX],
)
def export_model_to_onnx(model_output: Output[Artifact]):
    """Download TrOCR from HuggingFace and export to ONNX."""
    import os
    from optimum.exporters.onnx import main_export
    from transformers import TrOCRProcessor

    model_name = "microsoft/trocr-base-printed"
    output_dir = model_output.path
    os.makedirs(output_dir, exist_ok=True)

    main_export(
        model_name_or_path=model_name,
        output=output_dir,
        task="image-to-text",
        monolith=True,
    )

    processor = TrOCRProcessor.from_pretrained(model_name)
    processor.save_pretrained(output_dir)

    print(f"Model exported to {output_dir}")
    for f in os.listdir(output_dir):
        print(f"  {f}")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["boto3==1.42.83"],
    pip_index_urls=[EXTRA_INDEX],
)
def upload_to_s3(
    model_input: Input[Artifact],
    s3_endpoint: str,
    s3_bucket: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_folder: str = "pipelines/ocr-model/",
):
    """Upload ONNX model files to S3."""
    import os
    import boto3
    import botocore

    session = boto3.session.Session(
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )
    s3_resource = session.resource(
        "s3",
        config=botocore.client.Config(signature_version="s3v4"),
        endpoint_url=s3_endpoint,
        region_name="us-east-1",
        verify=False,
    )
    bucket = s3_resource.Bucket(s3_bucket)

    local_dir = model_input.path
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, local_dir)
            s3_key = os.path.join(s3_folder, relative_path)
            print(f"Uploading {file_path} -> {s3_key}")
            bucket.upload_file(file_path, s3_key)

    print("Upload complete.")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "boto3==1.42.83",
        "onnxruntime",
        "transformers==4.57.6",
        "pillow==12.2.0",
        "numpy",
    ],
    pip_index_urls=[EXTRA_INDEX],
)
def run_inference_from_s3(
    s3_endpoint: str,
    s3_bucket: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_folder: str = "pipelines/ocr-model/",
) -> str:
    """Download model from S3 and run OCR inference on a test image."""
    import os
    import boto3
    import botocore
    import numpy as np
    import onnxruntime as ort
    from transformers import TrOCRProcessor
    from PIL import Image

    # Download test image from S3
    s3_client = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        verify=False,
    )
    s3_client.download_file(s3_bucket, "pipelines/test-images/photo.jpg", "/tmp/test_image.jpg")
    print("Downloaded test image from S3")

    # Download model from S3
    working_folder = "/tmp/model/"
    os.makedirs(working_folder, exist_ok=True)

    session = boto3.session.Session(
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )
    s3_resource = session.resource(
        "s3",
        config=botocore.client.Config(signature_version="s3v4"),
        endpoint_url=s3_endpoint,
        region_name="us-east-1",
        verify=False,
    )
    bucket = s3_resource.Bucket(s3_bucket)
    s3_client = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        verify=False,
    )

    for obj in bucket.objects.filter(Prefix=s3_folder):
        file_name = obj.key.split("/")[-1]
        if not file_name:
            continue
        local_path = os.path.join(working_folder, file_name)
        print(f"Downloading {obj.key} -> {local_path}")
        s3_client.download_file(s3_bucket, obj.key, local_path)

    # Pre-processing
    image = Image.open("/tmp/test_image.jpg").convert("RGB")
    processor = TrOCRProcessor.from_pretrained(working_folder)
    pixel_values = processor(images=image, return_tensors="np").pixel_values
    batch_size = pixel_values.shape[0]

    # Load ONNX model
    model_path = os.path.join(working_folder, "model.onnx")
    ort_session = ort.InferenceSession(model_path)

    # Inference
    start_token_id = 2
    eos_token_id = 2
    max_length = 10
    generated_sequence = np.full((batch_size, 1), start_token_id, dtype=np.int64)

    for step in range(max_length):
        outputs = ort_session.run(
            None,
            {
                "pixel_values": pixel_values,
                "decoder_input_ids": generated_sequence,
            },
        )
        next_token_logits = outputs[0][:, -1, :]
        next_token_id = np.argmax(next_token_logits, axis=-1).reshape(
            (batch_size, 1)
        )
        generated_sequence = np.concatenate(
            [generated_sequence, next_token_id], axis=1
        )
        if np.any(next_token_id == eos_token_id):
            break

    # Post-processing
    generated_text = processor.batch_decode(
        generated_sequence, skip_special_tokens=True
    )[0]
    print(f">>> OCR Result <<<: {generated_text}")
    return generated_text


@dsl.pipeline(name="ocr_pipeline", description="Export TrOCR to ONNX, upload to S3, run inference")
def ocr_pipeline(
    s3_endpoint: str = "https://minio-s3-minio.apps.cluster-f44gq.f44gq.sandbox358.opentlc.com",
    s3_bucket: str = "ocr-demo",
    s3_access_key: str = "minio",
    s3_secret_key: str = "minio123",
):
    export_task = export_model_to_onnx()

    upload_task = upload_to_s3(
        model_input=export_task.outputs["model_output"],
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )

    inference_task = run_inference_from_s3(
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    inference_task.after(upload_task)


if __name__ == "__main__":
    compiler.Compiler().compile(ocr_pipeline, "ocr_pipeline.yaml")
    print("Pipeline compiled to ocr_pipeline.yaml")
