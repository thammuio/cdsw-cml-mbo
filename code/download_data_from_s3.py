import boto3
import botocore

def download_file_from_s3(bucket_name, file_key, local_path, aws_access_key_id, aws_secret_access_key):
    """
    Download a file from AWS S3 using access credentials.

    Parameters:
        bucket_name (str): The name of the S3 bucket.
        file_key (str): The object key (path) of the file in the S3 bucket.
        local_path (str): The local path where the file will be downloaded.
        aws_access_key_id (str): AWS Access Key ID.
        aws_secret_access_key (str): AWS Secret Access Key.

    Returns:
        bool: True if the file is successfully downloaded, False otherwise.
    """
    try:
        # Create a Boto3 S3 client with the provided access credentials
        s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

        # Download the file from S3 to the local path
        s3_client.download_file(bucket_name, file_key, local_path)

        print(f"File downloaded successfully to: {local_path}")
        return True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            print(f"Error downloading file: {e}")
        return False

# Usage example
if __name__ == "__main__":
    # Replace 'your-bucket-name', 'your-file-key', 'your-access-key', and 'your-secret-key'
    bucket_name = 'cc-fraud-data-kaggle'
    file_key = 'cc-fraud-data/c4cf0bee-c146-4da8-b6c0-4be053f5a060'
    local_file_path = 'data/data/creditcard.csv'  # Replace with the local path and filename
    aws_access_key_id = 'AKIAT45W6OH5K76LSB6N'
    aws_secret_access_key = 'IoleJNTesCFfh8L1K0x6/L0+q6rVITK5lPs1+VhC'

    download_file_from_s3(bucket_name, file_key, local_file_path, aws_access_key_id, aws_secret_access_key)
