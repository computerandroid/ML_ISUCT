import json
import logging
import time
import base64
import cv2
import numpy as np
from typing import Any, Dict, List
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable
from tritonclient.utils import InferenceServerException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonClient:
    def __init__(self, url: str) -> None:
        import tritonclient.grpc as grpcclient

        try:
            self.client = grpcclient.InferenceServerClient(url=url)
            self.brain_tumour_classes = {0: "no_tumour", 1: "tumour"}
        except ConnectionError as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing Triton client: {e}")
            raise

    def load_and_preprocess_image_from_bytes(
        self, image_bytes: bytes
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        try:
            img_array = np.frombuffer(image_bytes, np.uint8)
            img_original = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_original is None:
                raise ValueError("Could not decode image from bytes")
            
            img: np.ndarray[Any, np.dtype[Any]] = cv2.cvtColor(
                img_original, cv2.COLOR_BGR2RGB
            ).astype(np.uint8)
            img = cv2.resize(img, (224, 224)).astype(np.uint8)
            img = img.astype(np.float32)
            img = img / np.float32(255.0)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = ((img - mean) / std).astype(np.float32)
            img = np.transpose(img, (2, 0, 1))
            batch_data = np.expand_dims(img, axis=0).astype(np.float32)
            return batch_data
            
        except cv2.error as e:
            raise ValueError(f"OpenCV error during image processing: {e}")
        except Exception as e:
            raise RuntimeError(f"Image preprocessing failed: {e}")

    def predict_classification(
        self, input_data: np.ndarray[Any, np.dtype[np.float32]]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        import tritonclient.grpc as grpcclient

        try:
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
                
            inputs = [grpcclient.InferInput("input", input_data.shape, "FP32")]
            inputs[0].set_data_from_numpy(input_data)
            outputs = [grpcclient.InferRequestedOutput("output")]
            
            response = self.client.infer(
                model_name="brain_tumour_classifier", inputs=inputs, outputs=outputs
            )
            
            result_raw = response.as_numpy("output")
            if result_raw is None:
                raise ValueError("No result from Triton inference")
                
            result: np.ndarray[Any, np.dtype[np.float32]] = np.squeeze(result_raw).astype(
                np.float32
            )
            return result
            
        except InferenceServerException as e:
            raise RuntimeError(f"Triton inference error: {e}")
        except ValueError as e:
            raise ValueError(f"Data validation error: {e}")

    def postprocess_predictions(
        self, predictions: np.ndarray[Any, np.dtype[np.float32]]
    ) -> List[Dict[str, Any]]:
        try:
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()

            if len(predictions) == 2:
                exp_preds = np.exp(predictions - np.max(predictions))
                probabilities = exp_preds / np.sum(exp_preds)
            else:
                probability_tumour = 1 / (1 + np.exp(-predictions[0]))
                probabilities = np.array([1 - probability_tumour, probability_tumour])

            results: List[Dict[str, Any]] = []

            for class_id in [0, 1]:
                if class_id < len(probabilities):
                    prob = float(probabilities[class_id])
                    class_name = self.brain_tumour_classes.get(
                        class_id, f"class_{class_id}"
                    )

                    results.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": prob,
                            "diagnosis": "tumour" if class_id == 1 else "no_tumour",
                        }
                    )

            results.sort(key=lambda x: x["confidence"], reverse=True)
            return results
            
        except Exception as e:
            raise RuntimeError(f"Prediction postprocessing failed: {e}")


class BrainTumourClassificationConsumer:
    def __init__(self, bootstrap_servers: str = "kafka:9092") -> None:
        self.bootstrap_servers = bootstrap_servers
        self.consumer = self._create_consumer()
        self.result_producer = self._create_result_producer()
        self.triton_client = TritonClient("triton:8001")

    def _create_consumer(self) -> KafkaConsumer:
        try:
            return KafkaConsumer(
                "brain_mri_images",
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                api_version=(2, 8, 1),
                group_id="brain_tumour_consumers",
                auto_offset_reset="earliest",
            )
        except NoBrokersAvailable as e:
            logger.error(f"No Kafka brokers available: {e}")
            raise
        except KafkaError as e:
            logger.error(f"Kafka consumer creation failed: {e}")
            raise

    def _create_result_producer(self) -> KafkaProducer:
        try:
            return KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode("utf-8"),
                api_version=(2, 8, 1),
            )
        except NoBrokersAvailable as e:
            logger.error(f"No Kafka brokers available: {e}")
            raise
        except KafkaError as e:
            logger.error(f"Kafka producer creation failed: {e}")
            raise

    def save_result_to_kafka(self, result: Dict[str, Any]) -> None:
        try:
            self.result_producer.send("tumour_classification_results", value=result)
            self.result_producer.flush()
            logger.info(f"Result sent to Kafka: {result['data_id']}")
        except KafkaError as e:
            logger.error(f"Failed to send result to Kafka: {e}")
            raise

    def create_classification_result(
        self, data_id: str, predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {
            "data_id": data_id,
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else {},
            "processed_at": time.time(),
        }

    def process_message(self, mri_data: Dict[str, Any]) -> None:
        try:
            image_bytes = base64.b64decode(mri_data["image_data"])

            input_data = self.triton_client.load_and_preprocess_image_from_bytes(
                image_bytes
            )
            predictions = self.triton_client.predict_classification(input_data)
            prediction_results = self.triton_client.postprocess_predictions(predictions)
            
            if not prediction_results:
                logger.error(f"No predictions generated for {mri_data['data_id']}")
                return
                
            classification_result = self.create_classification_result(
                mri_data["data_id"], prediction_results
            )
            self.save_result_to_kafka(classification_result)

            top_pred = prediction_results[0]
            logger.info(
                f"Processed {mri_data['data_id']}: "
                f"{top_pred['diagnosis']} - "
                f"{top_pred['class_name']} ({top_pred['confidence']:.2%})"
            )
            
        except ValueError as e:
            logger.error(f"Data error processing {mri_data['data_id']}: {e}")
        except RuntimeError as e:
            logger.error(f"Runtime error processing {mri_data['data_id']}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing {mri_data['data_id']}: {e}")

    def process_kafka_messages(self) -> None:
        logger.info("Starting Brain Tumour Classification Consumer")
        for message in self.consumer:
            try:
                mri_data = message.value
                logger.info(f"Received MRI data: {mri_data['data_id']}")
                self.process_message(mri_data)
            except KeyError as e:
                logger.error(f"Missing required field in message: {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")


def main() -> None:
    consumer = BrainTumourClassificationConsumer()
    consumer.process_kafka_messages()


if __name__ == "__main__":
    main()
