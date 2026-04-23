import logging
import os
import socket
import time
import boto3
import numpy as np
import uvicorn  # pylint: disable=import-error
import cv2  # pylint: disable=import-error
from typing import Any, Dict, List
import tritonclient.grpc as grpcclient  # pylint: disable=import-error
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

TRITON_HOST = os.getenv("TRITON_HOST", "triton-server")
TRITON_PORT = os.getenv("TRITON_PORT", "8001")
MODEL_NAME = os.getenv("MODEL_NAME", "brain-cancer")
MODEL_INPUT_SHAPE = (244, 244, 3)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "brain-images")

app.state.triton_client = None
app.state.s3_client = None

def resolve_host(hostname: str) -> str:
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return hostname


def init_triton_client() -> bool:
    retries = 15
    for i in range(retries):
        try:
            host = resolve_host(TRITON_HOST)
            url = "%s:%s" % (host, TRITON_PORT)
            logger.info("Connecting to Triton at %s, attempt %d/%d", url, i + 1, retries)
            app.state.triton_client = grpcclient.InferenceServerClient(url=url, verbose=False)
            if app.state.triton_client.is_server_live():
                if app.state.triton_client.is_model_ready(MODEL_NAME):
                    logger.info("Triton server and model ready")
                    return True
            logger.warning("Triton not ready, retrying")
        except grpcclient.InferenceServerException as e:
            logger.warning("Triton init attempt %d failed: %s", i + 1, e)
        if i < retries - 1:
            time.sleep(3)
    logger.error("Failed to initialize Triton client")
    return False


def init_s3_client() -> bool:
    try:
        app.state.s3_client = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
        )
        app.state.s3_client.head_bucket(Bucket=BUCKET_NAME)
        return True
    except ClientError as e:
        logger.error("MinIO client init error: %s", e)
        return False


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # pylint: disable=no-member
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def convert_to_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # pylint: disable=no-member
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # pylint: disable=no-member
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
    return img


def resize_and_normalize(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (MODEL_INPUT_SHAPE[1], MODEL_INPUT_SHAPE[0]))  # pylint: disable=no-member
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def preprocess_image(data: bytes) -> np.ndarray:
    img = decode_image(data)
    img = convert_to_rgb(img)
    img = resize_and_normalize(img)
    return img


def infer(image: np.ndarray) -> List[Dict[str, Any]]:
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput("input__0", image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image)
    outputs.append(grpcclient.InferRequestedOutput("output__0"))
    results = app.state.triton_client.infer(MODEL_NAME, inputs=inputs, outputs=outputs)
    output_data = results.as_numpy("output__0")
    if output_data is None:
        raise RuntimeError("No output from model")
    predictions = []
    for idx, prob in enumerate(output_data[0]):
        predictions.append({"class_id": idx, "probability": float(prob)})
    return predictions


@app.on_event("startup")
async def startup_event():
    if not init_triton_client():
        logger.error("Triton client initialization failed on startup")
    if not init_s3_client():
        logger.error("S3 client initialization failed on startup")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = preprocess_image(content)
        preds = infer(image)
        return JSONResponse(content={"predictions": preds})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re)) from re
    except grpcclient.InferenceServerException as e:
        logger.error("Inference error: %s", e)
        raise HTTPException(status_code=500, detail="Inference failed") from e
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/health")
def health_check():
    if app.state.triton_client and app.state.triton_client.is_server_live():
        return {"status": "ok"}
    return JSONResponse(status_code=503, content={"status": "unavailable"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
