"""
Deploy the EventBridge-scheduled orchestrator Lambda
(aws/eventbridge/scheduler_handler.py).

This is the "fully autonomous" piece of the project: once deployed,
EventBridge invokes the orchestrator on a fixed schedule with zero
human involvement, and it runs the real agent (Claude + the 5 MCP
tools) against every machine, persisting traces to S3.

This script is a documented reference — it requires real AWS
credentials, an IAM role, an S3 bucket for traces, and your
ANTHROPIC_API_KEY, none of which exist in a sandbox. Review it and
run it from an environment with valid AWS credentials.

IMPORTANT — packaging note: this orchestrator imports the FULL agent
stack (langchain, langchain-anthropic, anthropic, httpx) which is
considerably heavier than the lightweight MCP servers in aws/lambda/.
This script builds its own deployment zip (not aws/lambda/package.zip)
for that reason — see build_orchestrator_package() below. Only agent/
and data/store.py are bundled as application source; verified
empirically that the orchestrator's import chain never touches
models/ or mcp_servers/ (it talks to sensor-mcp over HTTP, like
agent/tools.py always has, rather than importing the model directly).

Prerequisites:
    1. An S3 bucket for trace storage
    2. An IAM role with: AWSLambdaBasicExecutionRole + s3:PutObject on
       that bucket
    3. ANTHROPIC_API_KEY available to put into the function's environment
       (for a portfolio project, a plain env var is acceptable; for
       anything beyond that, pull it from AWS Secrets Manager at cold
       start instead and don't pass it as a plaintext Lambda env var)
    4. pip install boto3

Usage:
    python aws/eventbridge/deploy.py \\
        --role-arn arn:aws:iam::123456789012:role/LambdaExecutionRole \\
        --trace-bucket your-bucket-name \\
        --anthropic-api-key sk-ant-...
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
EVENTBRIDGE_DIR = Path(__file__).parent
ORCHESTRATOR_PACKAGE = EVENTBRIDGE_DIR / "orchestrator_package.zip"

FUNCTION_NAME = "asset-health-orchestrator"
RULE_NAME = "asset-health-scheduled-assessment"
SCHEDULE_EXPRESSION = "rate(15 minutes)"


def build_orchestrator_package() -> Path:
    """
    Package the orchestrator Lambda. Unlike aws/lambda/package.py
    (which is intentionally minimal — just FastAPI/Mangum/pydantic),
    this one needs the full agent stack: langchain, langchain-anthropic,
    anthropic, httpx, plus this project's own agent/, data/, and
    mcp_servers/ source (run_agent imports from all three).

    Dependencies must be pre-installed into EVENTBRIDGE_DIR / "_build"
    the same way aws/lambda/package.py expects — see that script's
    docstring for the exact pip install command shape, against:
        langchain, langchain-anthropic, anthropic, httpx, python-dotenv

    Only agent/ and data/store.py are bundled as application source —
    verified empirically that the orchestrator's actual import chain
    (data.store.get_all_machine_ids + agent.react_agent.run_agent)
    never touches models/ or mcp_servers/: agent/tools.py talks to
    sensor-mcp over HTTP rather than importing the model directly, and
    the orchestrator doesn't run any MCP server code itself.
    """
    build_dir = EVENTBRIDGE_DIR / "_build"
    if not build_dir.exists():
        print(
            f"Warning: {build_dir} doesn't exist — packaging application "
            f"code only, with NO dependencies. Install langchain, "
            f"langchain-anthropic, anthropic, httpx, python-dotenv into "
            f"{build_dir} first (see this script's docstring)."
        )

    staging = EVENTBRIDGE_DIR / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    if build_dir.exists():
        for item in build_dir.iterdir():
            dest = staging / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy(item, dest)

    # Application code the orchestrator's import chain actually needs.
    # Verified empirically (not just by reading imports) by tracking
    # sys.modules before/after calling the real import chain:
    #   from data.store import get_all_machine_ids
    #   from agent.react_agent import run_agent
    # → only agent/* and data/store.py actually load. agent/tools.py
    # talks to sensor-mcp over HTTP rather than importing models/
    # directly, and the orchestrator never touches mcp_servers/ at all
    # (that's the deployed Lambda/container code, not something this
    # process imports). Bundling those two directories would have
    # added dead weight with zero functional benefit.
    for module_dir in ["agent", "data"]:
        src = ROOT / module_dir
        dest = staging / module_dir
        shutil.copytree(
            src, dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "traces", "*.db"),
        )

    shutil.copy(EVENTBRIDGE_DIR / "scheduler_handler.py", staging / "scheduler_handler.py")

    with zipfile.ZipFile(ORCHESTRATOR_PACKAGE, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging))

    shutil.rmtree(staging)

    size_mb = ORCHESTRATOR_PACKAGE.stat().st_size / (1024 * 1024)
    print(f"Packaged orchestrator -> {ORCHESTRATOR_PACKAGE} ({size_mb:.2f} MB)")
    if size_mb > 50:
        print("Note: exceeds Lambda's 50MB direct-upload limit — upload to S3 instead.")
    return ORCHESTRATOR_PACKAGE


def deploy_lambda(lambda_client, role_arn: str, trace_bucket: str, anthropic_api_key: str) -> str:
    from botocore.exceptions import ClientError

    zip_bytes = ORCHESTRATOR_PACKAGE.read_bytes()
    env_vars = {"TRACE_BUCKET": trace_bucket, "ANTHROPIC_API_KEY": anthropic_api_key}

    try:
        response = lambda_client.create_function(
            FunctionName=FUNCTION_NAME,
            Runtime="python3.12",
            Role=role_arn,
            Handler="scheduler_handler.handler",
            Code={"ZipFile": zip_bytes},
            Environment={"Variables": env_vars},
            Timeout=300,       # 5 min — running the agent across 3 machines, each
                               # potentially making several Claude + tool round-trips
            MemorySize=512,
            Publish=True,
        )
        print(f"Created Lambda function: {FUNCTION_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            print(f"{FUNCTION_NAME} already exists — updating code and config")
            lambda_client.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=zip_bytes)
            response = lambda_client.update_function_configuration(
                FunctionName=FUNCTION_NAME, Environment={"Variables": env_vars}
            )
        else:
            raise

    return response["FunctionArn"]


def deploy_eventbridge_rule(events_client, lambda_client, function_arn: str):
    print(f"Creating EventBridge rule: {RULE_NAME} ({SCHEDULE_EXPRESSION})")
    rule_response = events_client.put_rule(
        Name=RULE_NAME,
        ScheduleExpression=SCHEDULE_EXPRESSION,
        State="ENABLED",
        Description="Triggers the agent orchestrator Lambda to assess all machines on a fixed interval",
    )

    events_client.put_targets(
        Rule=RULE_NAME,
        Targets=[{"Id": "asset-health-orchestrator", "Arn": function_arn}],
    )

    from botocore.exceptions import ClientError
    try:
        lambda_client.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId="AllowEventBridgeInvoke",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=rule_response["RuleArn"],
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceConflictException":
            raise

    print(f"EventBridge rule active: {rule_response['RuleArn']}")


def main():
    parser = argparse.ArgumentParser(description="Deploy the EventBridge-scheduled agent orchestrator")
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--trace-bucket", required=True)
    parser.add_argument("--anthropic-api-key", required=True)
    parser.add_argument("--skip-package", action="store_true",
                         help="Skip rebuilding orchestrator_package.zip (use existing one)")
    args = parser.parse_args()

    try:
        import boto3
    except ImportError:
        print("boto3 is required: pip install boto3", file=sys.stderr)
        sys.exit(1)

    if not args.skip_package:
        build_orchestrator_package()
    elif not ORCHESTRATOR_PACKAGE.exists():
        print(f"{ORCHESTRATOR_PACKAGE} not found and --skip-package was set.", file=sys.stderr)
        sys.exit(1)

    lambda_client = boto3.client("lambda")
    events_client = boto3.client("events")

    function_arn = deploy_lambda(lambda_client, args.role_arn, args.trace_bucket, args.anthropic_api_key)
    deploy_eventbridge_rule(events_client, lambda_client, function_arn)

    print(f"\nDone. {FUNCTION_NAME} will now run automatically every 15 minutes.")
    print(f"Traces will be written to s3://{args.trace_bucket}/traces/<machine_id>.json")


if __name__ == "__main__":
    main()
