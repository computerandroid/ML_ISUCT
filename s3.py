import boto3
import os
import time
import logging
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "brain-images")

def init_minio_client():
    """Initialize MinIO/S3 client with retry logic"""
    max_retries = 10
    for i in range(max_retries):
        try:
            s3_client = boto3.client(
                "s3",
                endpoint_url=MINIO_ENDPOINT,
                aws_access_key_id=MINIO_ACCESS_KEY,
                aws_secret_access_key=MINIO_SECRET_KEY,
                verify=False
            )
            
            # Test connection
            s3_client.list_buckets()
            logger.info("Successfully connected to MinIO")
            return s3_client
            
        except Exception as e:
            logger.warning(f"MinIO connection attempt {i+1}/{max_retries} failed: {e}")
            if i < max_retries - 1:
                time.sleep(5)
    
    raise Exception("Failed to connect to MinIO after all retries")

def create_bucket_if_not_exists(s3_client, bucket_name):
    """Create bucket if it doesn't exist"""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' already exists")
        return False
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            try:
                s3_client.create_bucket(Bucket=bucket_name)
                logger.info(f"Bucket '{bucket_name}' created successfully")
                bucket_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": "*",
                            "Action": [
                                "s3:GetObject",
                                "s3:ListBucket"
                            ],
                            "Resource": [
                                f"arn:aws:s3:::{bucket_name}",
                                f"arn:aws:s3:::{bucket_name}/*"
                            ]
                        }
                    ]
                }
                
                s3_client.put_bucket_policy(
                    Bucket=bucket_name,
                    Policy=json.dumps(bucket_policy)
                )
                logger.info(f"Bucket policy set for '{bucket_name}'")
                
                return True
            except Exception as create_error:
                logger.error(f"Failed to create bucket '{bucket_name}': {create_error}")
                raise
        else:
            logger.error(f"Error checking bucket '{bucket_name}': {e}")
            raise

def setup_folders(s3_client, bucket_name):
    """Create necessary folder structure in the bucket"""
    folders = ["original/", "processed/", "meta/", "results/"]
    
    for folder in folders:
        try:
            s3_client.put_object(Bucket=bucket_name, Key=folder)
            logger.info(f"Folder '{folder}' created in bucket '{bucket_name}'")
        except Exception as e:
            logger.warning(f"Failed to create folder '{folder}': {e}")

if __name__ == "__main__":
    import json
    
    logger.info("Starting S3 bucket initialization service...")
    
    try:
        time.sleep(10)
        
        # Initialize client
        s3_client = init_minio_client()
        
        # Create bucket
        created = create_bucket_if_not_exists(s3_client, BUCKET_NAME)
        
        # Setup folder structure
        setup_folders(s3_client, BUCKET_NAME)
        
        logger.info("S3 bucket initialization completed successfully")
        while True:
            time.sleep(30)
            try:
                s3_client.head_bucket(Bucket=BUCKET_NAME)
            except Exception as e:
                logger.error(f"Bucket health check failed: {e}")
                
    except Exception as e:
        logger.error(f"S3 initialization failed: {e}")
        exit(1)
