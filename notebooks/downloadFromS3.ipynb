#Upgrade pip to the latest version
!pip3 install --upgrade pip
#Install Boto3
!pip3 install boto3
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip3 install transformers
!pip install --upgrade jupyter ipywidgets


#Install Boto3 libraries
import os
import boto3
from botocore.client import Config
from boto3 import session
#Check Boto3 version
!pip3 show boto3


#Creating an S3 client
#Define credentials
key_id = os.environ.get('AWS_ACCESS_KEY_ID')
secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
endpoint = os.environ.get('AWS_S3_ENDPOINT')
region = os.environ.get('AWS_DEFAULT_REGION')
#Define client session
session = boto3.session.Session(aws_access_key_id=key_id,aws_secret_access_key=secret_key)

#Define client connection
s3_client = boto3.client('s3', aws_access_key_id=key_id,
aws_secret_access_key=secret_key,aws_session_token=None,
config=boto3.session.Config(signature_version='s3v4'),
endpoint_url=endpoint,
region_name=region)

s3_client.list_buckets()

# s3_client.download_file('prasad','Llama-3.2-3B-Instruct/','Llama-3.2-3B-Instruct/')

def download_directory_from_s3(bucket_name, s3_directory, local_directory):
    paginator = s3_client.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_directory):
        if 'Contents' not in result:
            continue

        for obj in result['Contents']:
            s3_key = obj['Key']
            local_file_path = os.path.join(local_directory, os.path.relpath(s3_key, s3_directory))

            if not os.path.exists(os.path.dirname(local_file_path)):
                os.makedirs(os.path.dirname(local_file_path))

            s3_client.download_file(bucket_name, s3_key, local_file_path)
            print(f'Downloaded {s3_key} to {local_file_path}')

# Example usage
bucket_name = 'prasad'
s3_directory = 'Llama-3.2-3B-Instruct/'

download_path = '/tmp/Llama-3.2-3B-Instruct/'

download_directory_from_s3(bucket_name, s3_directory, download_path)

