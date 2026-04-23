import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from typing import Any, Dict
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.transfers.local_to_s3 import (
    LocalFilesystemToS3Operator,
)
from airflow.utils.dates import days_ago
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from vit_pytorch import ViT


def download_processed_data() -> None:
    hook = S3Hook(aws_conn_id="aws_default")
    hook.download_file(
        key="processed/train_metadata.pkl",
        bucket_name="brain-tumor-bucket",
        local_path="/tmp/train_metadata.pkl",
    )
    hook.download_file(
        key="processed/test_metadata.pkl",
        bucket_name="brain-tumor-bucket",
        local_path="/tmp/test_metadata.pkl",
    )


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


class BrainTumorDataset:
    def __init__(self, dataset_dir: str) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_loaded = False

    def load_dataset(self) -> None:
        data_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.train_dataset = ImageFolder(
            os.path.join(self.dataset_dir, "Training"), transform=data_transforms
        )
        self.val_dataset = ImageFolder(
            os.path.join(self.dataset_dir, "Testing"), transform=data_transforms
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=32, shuffle=True
        )
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.dataset_loaded = True


def train_tumor_model(epochs: int = 100) -> None:
    dataset = BrainTumorDataset("/tmp")
    dataset.load_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TumorClassifierViT(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(
            dataset.train_loader, desc=f"Epoch {epoch+1}/{epochs}"
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataset.val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(dataset.val_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "/tmp/brain_tumor_model.pth")

    metrics = {
        "val_accuracy": val_accuracy,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "best_val_accuracy": best_val_accuracy,
    }

    with open("/tmp/metrics_tumor.json", "w") as f:
        json.dump(metrics, f)


default_args: Dict[str, Any] = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "retries": 3,
}

with DAG(
    "training_tumor",
    default_args=default_args,
    description="Brain Tumor Classification Training DAG",
    schedule_interval=None,
    catchup=False,
) as dag:
    data_sensor = S3KeySensor(
        task_id="data_sensor",
        bucket_key="processed/train_metadata.pkl",
        bucket_name="brain-tumor-bucket",
        aws_conn_id="aws_default",
        timeout=300,
        poke_interval=30,
    )

    download_data_op = PythonOperator(
        task_id="download_data_op",
        python_callable=download_processed_data,
    )

    train_model_op = PythonOperator(
        task_id="train_model_op",
        python_callable=lambda: train_tumor_model(epochs=100),
    )

    upload_model_op = LocalFilesystemToS3Operator(
        task_id="upload_model_op",
        filename="/tmp/brain_tumor_model.pth",
        dest_key="models/brain_tumor_model.pth",
        dest_bucket="brain-tumor-bucket",
        replace=True,
        aws_conn_id="aws_default",
    )

    upload_metrics_op = LocalFilesystemToS3Operator(
        task_id="upload_metrics_op",
        filename="/tmp/metrics_tumor.json",
        dest_key="metrics/metrics_tumor.json",
        dest_bucket="brain-tumor-bucket",
        replace=True,
        aws_conn_id="aws_default",
    )

    data_sensor >> download_data_op >> train_model_op
    train_model_op >> [upload_model_op, upload_metrics_op]
