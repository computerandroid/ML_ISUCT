import json
import logging
import time
import uuid
import base64
from typing import Dict, List, Union, cast
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError, NoBrokersAvailable, KafkaTimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TumourDataProducer:
    def __init__(self, bootstrap_servers: str = "kafka:9092") -> None:
        self.bootstrap_servers = bootstrap_servers
        self.producer = self._create_producer()
        self.topic = "tumour_images"

    def _create_producer(self) -> KafkaProducer:
        try:
            return KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode("utf-8"),
                api_version=(2, 8, 1),
            )
        except NoBrokersAvailable as e:
            logger.error("No Kafka brokers available: %s", e)
            raise
        except KafkaError as e:
            logger.error("Kafka producer creation failed: %s", e)
            raise

    def scan_images_from_kafka(self) -> List[Dict[str, Union[str, bytes]]]:
        try:
            consumer = KafkaConsumer(
                "raw_images",
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda x: x,
                api_version=(2, 8, 1),
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                consumer_timeout_ms=1000,
            )

            images_data = []
            for message in consumer:
                image_data = {
                    "image_data": message.value,
                    "filename": f"image_{message.offset}.jpg",
                }
                images_data.append(image_data)
                consumer.commit()
                break

            consumer.close()
            return images_data
            
        except KafkaError as e:
            logger.error("Kafka consumer error: %s", e)
            return []
        except Exception as e:
            logger.error("Unexpected error scanning images: %s", e)
            return []

    def create_tumour_data(
        self, image_data: bytes, filename: str
    ) -> Dict[str, Union[str, float]]:
        try:
            image_data_base64 = base64.b64encode(image_data).decode("utf-8")
            return {
                "data_id": str(uuid.uuid4()),
                "image_data": image_data_base64,
                "filename": filename,
                "timestamp": time.time(),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to create tumour data: {e}")

    def send_tumour_data(self, image_data: Dict[str, Union[str, bytes]]) -> None:
        try:
            image_bytes = cast(bytes, image_data["image_data"])
            filename = cast(str, image_data["filename"])

            tumour_data = self.create_tumour_data(image_bytes, filename)
            future = self.producer.send(
                self.topic,
                key=str(tumour_data["data_id"]).encode("utf-8"),
                value=tumour_data,
            )
            future.get(timeout=10)
            logger.info("Sent tumour data: %s", tumour_data["data_id"])
            
        except KafkaTimeoutError as e:
            logger.error("Kafka timeout sending message: %s", e)
        except KafkaError as e:
            logger.error("Kafka error sending message: %s", e)
        except RuntimeError as e:
            logger.error("Data creation error: %s", e)
        except Exception as e:
            logger.error("Unexpected error sending message: %s", e)

    def run(self) -> None:
        logger.info("Starting Tumour Data Producer")
        while True:
            try:
                image_files_data = self.scan_images_from_kafka()
                if image_files_data:
                    for image_data in image_files_data:
                        self.send_tumour_data(image_data)
                    logger.info("Processed %d images", len(image_files_data))
                else:
                    logger.info("No images found in Kafka topic")
                time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Producer stopped by user")
                break
            except Exception as e:
                logger.error("Error in producer: %s", e)
                time.sleep(5)


def main() -> None:
    producer = TumourDataProducer()
    producer.run()


if __name__ == "__main__":
    main()
