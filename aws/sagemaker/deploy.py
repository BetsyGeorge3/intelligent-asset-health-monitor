"""
Deploy the packaged BiLSTM model to a SageMaker real-time endpoint.

This script is provided as a documented reference for the deployment
step — it requires actual AWS credentials, an S3 bucket, and IAM
permissions to run, none of which exist in a local/sandbox environment.
Review it, fill in the placeholders, and run it from an environment
with valid AWS credentials configured (e.g. `aws configure`).

Prerequisites:
    1. python aws/sagemaker/package_model.py   (produces model.tar.gz)
    2. An S3 bucket you can write to
    3. An IAM role with SageMakerFullAccess (or a scoped-down equivalent)
       and a trust policy allowing sagemaker.amazonaws.com to assume it
    4. pip install boto3 sagemaker

Usage:
    python aws/sagemaker/deploy.py \\
        --bucket your-bucket-name \\
        --role-arn arn:aws:iam::123456789012:role/SageMakerExecutionRole
"""

import argparse
import sys
import time
from pathlib import Path

SAGEMAKER_DIR = Path(__file__).parent
MODEL_TARBALL = SAGEMAKER_DIR / "model.tar.gz"

ENDPOINT_NAME = "asset-health-bilstm-anomaly-detector"
MODEL_NAME = f"{ENDPOINT_NAME}-model"
ENDPOINT_CONFIG_NAME = f"{ENDPOINT_NAME}-config"

# SageMaker's official PyTorch inference container. Region-specific —
# this is the us-east-1 URI; swap the region in the URI if deploying
# elsewhere. See: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
PYTORCH_INFERENCE_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-inference:2.2.0-cpu-py310-ubuntu20.04-sagemaker"
)


def upload_model_to_s3(bucket: str, key: str = "asset-health/model.tar.gz") -> str:
    import boto3

    if not MODEL_TARBALL.exists():
        raise FileNotFoundError(
            f"{MODEL_TARBALL} not found — run `python aws/sagemaker/package_model.py` first."
        )

    s3 = boto3.client("s3")
    print(f"Uploading {MODEL_TARBALL} -> s3://{bucket}/{key} ...")
    s3.upload_file(str(MODEL_TARBALL), bucket, key)
    s3_uri = f"s3://{bucket}/{key}"
    print(f"Uploaded: {s3_uri}")
    return s3_uri


def deploy_endpoint(model_s3_uri: str, role_arn: str, instance_type: str = "ml.t2.medium"):
    import boto3

    sm = boto3.client("sagemaker")

    print(f"Creating SageMaker model '{MODEL_NAME}'...")
    sm.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={
            "Image": PYTORCH_INFERENCE_IMAGE,
            "ModelDataUrl": model_s3_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": model_s3_uri,
            },
        },
        ExecutionRoleArn=role_arn,
    )

    print(f"Creating endpoint config '{ENDPOINT_CONFIG_NAME}'...")
    sm.create_endpoint_config(
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": MODEL_NAME,
            "InstanceType": instance_type,
            "InitialInstanceCount": 1,
        }],
    )

    print(f"Creating endpoint '{ENDPOINT_NAME}' (this takes several minutes)...")
    sm.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
    )

    print("Waiting for endpoint to become InService...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=ENDPOINT_NAME)

    print(f"\nEndpoint '{ENDPOINT_NAME}' is now InService.")
    print(
        f"Point models/inference.py at this endpoint name to use it "
        f"instead of local weights (see the USE_SAGEMAKER_ENDPOINT note there)."
    )


def main():
    parser = argparse.ArgumentParser(description="Deploy the BiLSTM model to a SageMaker endpoint")
    parser.add_argument("--bucket", required=True, help="S3 bucket to upload model.tar.gz to")
    parser.add_argument("--role-arn", required=True, help="IAM role ARN with SageMaker execution permissions")
    parser.add_argument("--instance-type", default="ml.t2.medium", help="SageMaker instance type (default: ml.t2.medium)")
    args = parser.parse_args()

    try:
        import boto3  # noqa: F401
    except ImportError:
        print("boto3 is required: pip install boto3 sagemaker", file=sys.stderr)
        sys.exit(1)

    s3_uri = upload_model_to_s3(args.bucket)
    deploy_endpoint(s3_uri, args.role_arn, args.instance_type)


if __name__ == "__main__":
    main()
