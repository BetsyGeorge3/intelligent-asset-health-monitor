"""
Deploy the 4 lightweight MCP servers (cmms, inventory, scheduling,
notify) as separate Lambda functions, all sharing the one package.zip
built by package.py — each function differs only in its
MCP_SERVER_MODULE environment variable and its name.

This script is a documented reference — it requires real AWS
credentials, an IAM execution role, and (for HTTP access) an API
Gateway or Lambda Function URL, none of which exist in a sandbox.
Review it, fill in the role ARN, and run it from an environment with
valid AWS credentials configured.

Prerequisites:
    1. pip install --platform manylinux2014_x86_64 --target aws/lambda/_build \\
           --only-binary=:all: -r aws/lambda/requirements.txt
    2. python aws/lambda/package.py     (produces aws/lambda/package.zip)
    3. An IAM role with basic Lambda execution permissions (the
       AWSLambdaBasicExecutionRole managed policy is sufficient — none
       of these 4 servers touch other AWS services directly)
    4. pip install boto3

Usage:
    python aws/lambda/deploy.py --role-arn arn:aws:iam::123456789012:role/LambdaExecutionRole

Each function gets a Lambda Function URL (the simplest way to get an
HTTPS endpoint per function without provisioning API Gateway) — printed
at the end of the run. Update mcp_servers' base URLs (wherever
agent/tools.py or the dashboard reads MCP_BASE_URLS) to point at these
instead of localhost:800X once deployed.
"""

import argparse
import sys
from pathlib import Path

LAMBDA_DIR = Path(__file__).parent
PACKAGE_ZIP = LAMBDA_DIR / "package.zip"

# One function per lightweight server. sensor-mcp is intentionally
# absent — see handler.py's docstring for why it stays off Lambda.
FUNCTIONS = [
    {"name": "asset-health-cmms-mcp",       "module": "mcp_servers.cmms_mcp:app"},
    {"name": "asset-health-inventory-mcp",  "module": "mcp_servers.inventory_mcp:app"},
    {"name": "asset-health-scheduling-mcp", "module": "mcp_servers.scheduling_mcp:app"},
    {"name": "asset-health-notify-mcp",     "module": "mcp_servers.notify_mcp:app"},
]


def deploy_function(lambda_client, name: str, module: str, role_arn: str, zip_bytes: bytes) -> str:
    """Create (or update, if it already exists) one Lambda function. Returns its Function URL."""
    from botocore.exceptions import ClientError

    print(f"Deploying {name} (MCP_SERVER_MODULE={module})...")

    try:
        lambda_client.create_function(
            FunctionName=name,
            Runtime="python3.12",
            Role=role_arn,
            Handler="handler.handler",
            Code={"ZipFile": zip_bytes},
            Environment={"Variables": {"MCP_SERVER_MODULE": module}},
            Timeout=15,
            MemorySize=256,
            Publish=True,
        )
        print(f"  Created {name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            print(f"  {name} already exists — updating code and config instead")
            lambda_client.update_function_code(FunctionName=name, ZipFile=zip_bytes)
            lambda_client.update_function_configuration(
                FunctionName=name,
                Environment={"Variables": {"MCP_SERVER_MODULE": module}},
            )
        else:
            raise

    # Function URL — simplest way to get an HTTPS endpoint per function
    # without provisioning API Gateway separately.
    try:
        url_config = lambda_client.create_function_url_config(
            FunctionName=name,
            AuthType="NONE",  # NOTE: open endpoint — fine for a portfolio demo,
                               # switch to AWS_IAM auth for anything beyond that.
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            url_config = lambda_client.get_function_url_config(FunctionName=name)
        else:
            raise

    # Function URLs need explicit public-invoke permission the first time.
    try:
        lambda_client.add_permission(
            FunctionName=name,
            StatementId="FunctionURLAllowPublicAccess",
            Action="lambda:InvokeFunctionUrl",
            Principal="*",
            FunctionUrlAuthType="NONE",
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceConflictException":
            raise

    return url_config["FunctionUrl"]


def main():
    parser = argparse.ArgumentParser(description="Deploy the 4 lightweight MCP servers to Lambda")
    parser.add_argument("--role-arn", required=True, help="IAM role ARN for the Lambda functions")
    args = parser.parse_args()

    if not PACKAGE_ZIP.exists():
        print(f"{PACKAGE_ZIP} not found — run `python aws/lambda/package.py` first.", file=sys.stderr)
        sys.exit(1)

    try:
        import boto3
    except ImportError:
        print("boto3 is required: pip install boto3", file=sys.stderr)
        sys.exit(1)

    lambda_client = boto3.client("lambda")
    zip_bytes = PACKAGE_ZIP.read_bytes()

    print(f"Package size: {len(zip_bytes) / (1024*1024):.2f} MB\n")

    urls = {}
    for fn in FUNCTIONS:
        urls[fn["name"]] = deploy_function(lambda_client, fn["name"], fn["module"], args.role_arn, zip_bytes)

    print("\nAll 4 functions deployed. Function URLs:")
    for name, url in urls.items():
        print(f"  {name:<32} {url}")


if __name__ == "__main__":
    main()
