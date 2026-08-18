import boto3
import os
from src.constant.env_variable import AWS_SECRET_ACCESS_KEY_ENV_KEY, AWS_ACCESS_KEY_ID_ENV_KEY, REGION_NAME


class S3Client:

    s3_client = None
    s3_resource = None

    def __init__(self, region_name=REGION_NAME):

        if S3Client.s3_resource == None or S3Client.s3_client == None:
            __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)

            # NOTE: AWS_S3_ENDPOINT_URL points boto3 at Backblaze B2
            # (S3-compatible) instead of real AWS, since AWS/Azure
            # signup weren't accessible. If unset, boto3 falls back
            # to real AWS S3 normally.
            __endpoint_url = os.getenv("AWS_S3_ENDPOINT_URL")

            if __access_key_id is None:
                raise Exception(f"Environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY} is not set.")
            if __secret_access_key is None:
                raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set.")

            S3Client.s3_resource = boto3.resource(
                's3',
                aws_access_key_id=__access_key_id,
                aws_secret_access_key=__secret_access_key,
                region_name=region_name,
                endpoint_url=__endpoint_url,
            )
            S3Client.s3_client = boto3.client(
                's3',
                aws_access_key_id=__access_key_id,
                aws_secret_access_key=__secret_access_key,
                region_name=region_name,
                endpoint_url=__endpoint_url,
            )
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client
