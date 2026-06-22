"""
Lambda handler wrapper for the 4 lightweight MCP servers
(cmms, inventory, scheduling, notify).

AWS Lambda doesn't speak ASGI natively — it invokes a plain
`handler(event, context)` function with an API Gateway-shaped event
dict. Mangum bridges that gap: it wraps any ASGI app (our FastAPI
apps included) so the exact same `mcp_servers/*.py` code that runs
under uvicorn locally also runs unmodified inside Lambda.

Which of the 4 servers this particular Lambda function wraps is
decided by the MCP_SERVER_MODULE environment variable — same pattern
as docker/Dockerfile.lightweight, so one handler file serves all 4
Lambda functions (set via each function's env var at creation time,
see aws/lambda/deploy.py).

sensor-mcp is deliberately NOT included here — it needs torch + the
~580KB model weights, which makes it a poor fit for Lambda's package
size limits and cold-start characteristics. It stays on SageMaker
(aws/sagemaker/) for inference and would run on ECS/Fargate or EC2
for the HTTP layer in a real deployment, not Lambda.

Local testing (no AWS needed):
    python aws/lambda/handler.py
"""

import importlib
import os
import sys
from pathlib import Path

from mangum import Mangum

# Ensures `mcp_servers` is importable regardless of working directory.
# Locally this resolves to the project root (three levels up from this
# file); inside the Lambda deployment package (see aws/lambda/package.py)
# mcp_servers/ is copied to sit alongside this handler.py at the zip's
# root, so this insert is a harmless no-op there.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mirrors docker/Dockerfile.lightweight's MCP_SERVER_MODULE convention:
# "mcp_servers.cmms_mcp:app" → module "mcp_servers.cmms_mcp", attribute "app"
MODULE_PATH, APP_ATTR = os.environ.get(
    "MCP_SERVER_MODULE", "mcp_servers.cmms_mcp:app"
).split(":")

_module = importlib.import_module(MODULE_PATH)
app = getattr(_module, APP_ATTR)

# With lifespan="off" (below), FastAPI's lifespan context manager never
# runs — which means each server's init_db() (the function that creates
# its SQLite tables and seeds reference data) would never fire, and the
# very first request would crash with "no such table: ...". Call it
# explicitly here instead, once per cold start, which is the correct
# place for one-time setup work in a Lambda execution environment.
if hasattr(_module, "init_db"):
    _module.init_db()

# lifespan="off" — Lambda invocations are stateless and short-lived, and
# a fresh container gets a fresh /tmp (or EFS-mounted volume) on most
# cold starts anyway, so there's no ongoing FastAPI app state for a
# lifespan to manage here. init_db() above covers the one thing that
# actually mattered from the lifespan handler.
handler = Mangum(app, lifespan="off")


if __name__ == "__main__":
    # Quick local smoke test simulating an API Gateway proxy event,
    # without needing AWS credentials or a deployed function.
    fake_event = {
        "version": "2.0",
        "routeKey": "GET /health",
        "rawPath": "/health",
        "rawQueryString": "",
        "headers": {"host": "localhost"},
        "requestContext": {
            "http": {"method": "GET", "path": "/health", "sourceIp": "127.0.0.1"},
            "stage": "$default",
        },
        "isBase64Encoded": False,
    }

    class FakeContext:
        function_name = "local-test"
        memory_limit_in_mb = 128
        aws_request_id = "local-test-request-id"

    response = handler(fake_event, FakeContext())
    print("Lambda handler response:")
    print(response)
