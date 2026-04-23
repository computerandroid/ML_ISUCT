import os
import time
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable, KafkaTimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageLoader:
    def __init__(
        self, bootstrap_servers: str = "kafka:9092", image_folder: str = "/images"
    ):
        self.bootstrap_servers = bootstrap_servers
        self.image_folder = image_folder
        self.producer = self._create_producer()
        self.topic = "raw_images"

    def _create_producer(self) -> KafkaProducer:
        try:
            return KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                api_version=(2, 8, 1),
            )
        except NoBrokersAvailable as e:
            logger.error(f"No Kafka brokers available: {e}")
            raise
        except KafkaError as e:
            logger.error(f"Kafka producer creation failed: {e}")
            raise

    def load_images_to_kafka(self) -> int:
        if not os.path.exists(self.image_folder):
            logger.error(f"Image folder {self.image_folder} not found")
            return 0

        image_files = [
            f
            for f in os.listdir(self.image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            logger.warning(f"No images found in {self.image_folder}")
            return 0

        loaded_count = 0
        for image_file in image_files:
            try:
                image_path = os.path.join(self.image_folder, image_file)
                with open(image_path, "rb") as f:
                    image_data = f.read()

                future = self.producer.send(
                    self.topic, key=image_file.encode("utf-8"), value=image_data
                )
                future.get(timeout=10)
                logger.info(f"Loaded image to Kafka: {image_file}")
                loaded_count += 1

            except FileNotFoundError as e:
                logger.error(f"Image file not found: {e}")
            except KafkaTimeoutError as e:
                logger.error(f"Kafka timeout loading {image_file}: {e}")
            except KafkaError as e:
                logger.error(f"Kafka error loading {image_file}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading {image_file}: {e}")

        self.producer.flush()
        logger.info(
            f"Successfully loaded {loaded_count} images to Kafka topic: {self.topic}"
        )
        return loaded_count

    def run_continuous(self) -> None:
        logger.info(
            "Starting Image Loader - loading images from %s to topic: %s",
            self.image_folder,
            self.topic,
        )

        initial_count = self.load_images_to_kafka()
        if initial_count == 0:
            logger.warning("No images loaded initially. Waiting for images...")

        processed_files = set(os.listdir(self.image_folder))

        while True:
            try:
                time.sleep(10)

                current_files = set(os.listdir(self.image_folder))
                new_files = current_files - processed_files

                if new_files:
                    new_image_files = [
                        f
                        for f in new_files
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]

                    for image_file in new_image_files:
                        try:
                            image_path = os.path.join(self.image_folder, image_file)
                            with open(image_path, "rb") as f:
                                image_data = f.read()

                            future = self.producer.send(
                                self.topic,
                                key=image_file.encode("utf-8"),
                                value=image_data,
                            )
                            future.get(timeout=10)
                            logger.info(f"Loaded new image to Kafka: {image_file}")
                            processed_files.add(image_file)

                        except FileNotFoundError as e:
                            logger.error(f"New image file not found: {e}")
                        except KafkaTimeoutError as e:
                            logger.error(f"Kafka timeout loading new image {image_file}: {e}")
                        except KafkaError as e:
                            logger.error(f"Kafka error loading new image {image_file}: {e}")
                        except Exception as e:
                            logger.error(f"Unexpected error loading new image {image_file}: {e}")

            except KeyboardInterrupt:
                logger.info("Image Loader stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in Image Loader: {e}")
                time.sleep(5)


def main() -> None:
    loader = ImageLoader()
    loader.run_continuous()


if __name__ == "__main__":
    main()
