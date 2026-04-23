import json
import os
from typing import Any, Dict
import boto3
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.transfers.local_to_s3 import (
    LocalFilesystemToS3Operator,
)
from airflow.utils.dates import days_ago
from vit_pytorch import ViT


class TumorClassifierViT(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.vit = ViT(
            image_size=224,
            patch_size=32,
            num_classes=num_classes,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


def model_choose_op() -> str:
    s3_client = boto3.client("s3")
    try:
        obj = s3_client.get_object(
            Bucket="brain-tumor-bucket", Key="metrics/metrics_tumor.json"
        )
        metrics = json.loads(obj["Body"].read().decode("utf-8"))
        if metrics["val_accuracy"] > 0.85:
            return "models/brain_tumor_model.pth"
        else:
            return "models/backup_brain_tumor_model.pth"
    except Exception:
        return "models/brain_tumor_model.pth"


def inference_model_op() -> None:
    model_path = "/tmp/brain_tumor_model.pth"
    input_dir = "/tmp/inference_data"
    output_dir = "/tmp/inference_results"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TumorClassifierViT(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    results = []

    for img_file in os.listdir(input_dir):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(input_dir, img_file)
            try:
                image = Image.open(img_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    predicted_class = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_class].item()

                results.append(
                    {
                        "image": img_file,
                        "prediction": class_names[predicted_class],
                        "confidence": float(confidence),
                        "probabilities": {
                            class_names[i]: float(probabilities[i])
                            for i in range(len(class_names))
                        },
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "image": img_file,
                        "error": str(e),
                        "prediction": "unknown",
                        "confidence": 0.0,
                    }
                )

    with open(os.path.join(output_dir, "tumor_predictions.json"), "w") as f:
        json.dump(results, f, indent=2)


def download_inference_data() -> None:
    hook = S3Hook(aws_conn_id="aws_default")
    keys = hook.list_keys(bucket_name="brain-tumor-bucket", prefix="inference/")
    if keys:
        os.makedirs("/tmp/inference_data", exist_ok=True)
        for key in keys:
            if key.endswith("/"):
                continue
            local_path = "/tmp/inference_data/" + os.path.basename(key)
            hook.download_file(
                key=key, bucket_name="brain-tumor-bucket", local_path=local_path
            )


def download_model(model_key: str) -> None:
    hook = S3Hook(aws_conn_id="aws_default")
    hook.download_file(
        key=model_key,
        bucket_name="brain-tumor-bucket",
        local_path="/tmp/brain_tumor_model.pth",
    )


default_args: Dict[str, Any] = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "retries": 3,
}

with DAG(
    "inference_tumor",
    default_args=default_args,
    description="Brain Tumor Classification Inference DAG",
    schedule_interval=None,
    catchup=False,
) as dag:
    model_sensor = S3KeySensor(
        task_id="model_sensor",
        bucket_key="models/brain_tumor_model.pth",
        bucket_name="brain-tumor-bucket",
        aws_conn_id="aws_default",
        timeout=300,
        poke_interval=30,
    )

    data_sensor = S3KeySensor(
        task_id="data_sensor",
        bucket_key="inference/*.jpg",
        bucket_name="brain-tumor-bucket",
        aws_conn_id="aws_default",
        timeout=300,
        poke_interval=30,
    )

    model_choose_task = PythonOperator(
        task_id="model_choose_op",
        python_callable=model_choose_op,
    )

    download_data_op = PythonOperator(
        task_id="download_data_op",
        python_callable=download_inference_data,
    )

    download_model_op = PythonOperator(
        task_id="download_model_op",
        python_callable=lambda **kwargs: download_model(
            kwargs["task_instance"].xcom_pull(task_ids="model_choose_op")
        ),
        provide_context=True,
    )

    inference_model_task = PythonOperator(
        task_id="inference_model_op",
        python_callable=inference_model_op,
    )

    upload_result_op = LocalFilesystemToS3Operator(
        task_id="upload_result_op",
        filename="/tmp/inference_results/tumor_predictions.json",
        dest_key="results/tumor_predictions.json",
        dest_bucket="brain-tumor-bucket",
        replace=True,
        aws_conn_id="aws_default",
    )

    [model_sensor, data_sensor] >> model_choose_task
    model_choose_task >> download_model_op
    data_sensor >> download_data_op
    [download_model_op, download_data_op] >> inference_model_task >> upload_result_op
