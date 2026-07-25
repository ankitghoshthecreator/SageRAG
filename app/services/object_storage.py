import logging
from io import BytesIO
from typing import Optional
from app.utils.config import settings

logger = logging.getLogger("sagerag.storage")

class MinioStorage:
    def __init__(self):
        self.client = None
        self.bucket_name = settings.MINIO_BUCKET_NAME
        self._connect()

    def _connect(self):
        try:
            from minio import Minio
            self.client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )
            # Lightweight check to ensure connections are functional
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created MinIO bucket '{self.bucket_name}'")
            logger.info(f"Connected to MinIO at {settings.MINIO_ENDPOINT}")
        except Exception as e:
            logger.warning(
                f"Could not connect to MinIO ({e}). "
                "Files will be managed via the local disk upload directory only."
            )
            self.client = None

    def upload_file(self, object_name: str, file_data: bytes, content_type: str = "application/octet-stream") -> bool:
        """Upload file bytes to MinIO."""
        if not self.client:
            return False
        try:
            data_stream = BytesIO(file_data)
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=data_stream,
                length=len(file_data),
                content_type=content_type,
            )
            logger.info(f"Successfully uploaded {object_name} to MinIO")
            return True
        except Exception as e:
            logger.error(f"MinIO upload error for {object_name}: {e}")
            return False

    def download_file(self, object_name: str) -> Optional[bytes]:
        """Download file bytes from MinIO."""
        if not self.client:
            return None
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except Exception as e:
            logger.error(f"MinIO download error for {object_name}: {e}")
            return None

    def delete_file(self, object_name: str) -> bool:
        """Delete a file from MinIO."""
        if not self.client:
            return False
        try:
            self.client.remove_object(self.bucket_name, object_name)
            logger.info(f"Successfully deleted {object_name} from MinIO")
            return True
        except Exception as e:
            logger.error(f"MinIO delete error for {object_name}: {e}")
            return False

# Initialize the storage client instance
storage_client = MinioStorage()
