import os
import boto3
import botocore

def upload_files_to_s3(bucket_name, local_directory, s3_prefix, aws_access_key_id, aws_secret_access_key):
    """
    Upload all files from a local directory to AWS S3 using access credentials.

    Parameters:
        bucket_name (str): The name of the S3 bucket.
        local_directory (str): The local directory containing files to be uploaded.
        s3_prefix (str): The S3 object prefix (path) for the uploaded files.
        aws_access_key_id (str): AWS Access Key ID.
        aws_secret_access_key (str): AWS Secret Access Key.

    Returns:
        bool: True if all files are successfully uploaded, False otherwise.
    """
    try:
        # Create a Boto3 S3 client with the provided access credentials
        s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

        # List all files in the local directory
        for root, _, files in os.walk(local_directory):
            for filename in files:
                local_file_path = os.path.join(root, filename)
                s3_key = os.path.join(s3_prefix, filename)

                # Upload the file to S3
                s3_client.upload_file(local_file_path, bucket_name, s3_key)
                print(f"File uploaded successfully to: s3://{bucket_name}/{s3_key}")

        return True
    except botocore.exceptions.ClientError as e:
        print(f"Error uploading files: {e}")
        return False

# Usage example
if __name__ == "__main__":
    # Replace 'your-bucket-name', 'your-aws-access-key', 'your-aws-secret-key',
    # 'path/to/data/files', and 'cc-fraud-score/' with appropriate values
    bucket_name = 'cc-fraud-data-kaggle'
    aws_access_key_id = 'AKIAT45W6OH5K76LSB6N'
    aws_secret_access_key = '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'
    local_directory = 'data/score'
    s3_prefix = 'cc-fraud-score/'

    upload_files_to_s3(bucket_name, local_directory, s3_prefix, aws_access_key_id, aws_secret_access_key)
