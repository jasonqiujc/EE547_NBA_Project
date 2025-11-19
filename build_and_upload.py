# build_and_upload.py

from build_datasets import build_datasets
from s3_uploader import upload_files_to_s3


def main():
    print("Step 1: Generate local datasets ...")
    files = build_datasets()
    for f in files:
        print("  -", f)

    print("\nStep 2: Upload data to S3 (boto3 + IAM Role) ...")
    s3_keys = upload_files_to_s3(files)

    print("\nUploaded S3 keys:")
    for k in s3_keys:
        print("  -", k)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
