"""
Package each of the 4 lightweight MCP servers (cmms, inventory,
scheduling, notify) into a Lambda deployment zip.

All 4 share the exact same zip contents — the only difference between
their deployed Lambda functions is the MCP_SERVER_MODULE environment
variable set at function-creation time (see deploy.py). So this script
builds ONE deployment package and aws/lambda/deploy.py reuses it for
all 4 functions, rather than building 4 near-identical zips.

Package layout (matches what handler.py's sys.path insert + Mangum
import expect):
    package.zip
    ├── handler.py
    ├── mcp_servers/
    │   ├── __init__.py
    │   ├── cmms_mcp.py
    │   ├── inventory_mcp.py
    │   ├── scheduling_mcp.py
    │   ├── notify_mcp.py
    │   └── schemas.py
    └── <installed dependencies: fastapi, mangum, pydantic, etc.>

Usage:
    cd asset_health_monitor
    pip install -r aws/lambda/requirements.txt -t aws/lambda/_build/  # see note below
    python aws/lambda/package.py

Note on dependencies: this script does NOT install pip packages itself
(doing that correctly requires matching Lambda's exact runtime — e.g.
manylinux wheels — which isn't reliable to automate generically across
every machine this might run on). Install dependencies into
aws/lambda/_build/ first using a command appropriate for your target
Lambda runtime, e.g.:
    pip install --platform manylinux2014_x86_64 --target aws/lambda/_build \\
        --only-binary=:all: -r aws/lambda/requirements.txt
then run this script to fold in the application code on top.
"""

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
LAMBDA_DIR = Path(__file__).parent
BUILD_DIR = LAMBDA_DIR / "_build"

# Only what the 4 lightweight servers + handler.py actually need —
# deliberately excludes sensor_mcp.py (needs torch, stays off Lambda).
MCP_SERVER_FILES = [
    "__init__.py",
    "schemas.py",
    "cmms_mcp.py",
    "inventory_mcp.py",
    "scheduling_mcp.py",
    "notify_mcp.py",
]


def build_package(output_path: Path = LAMBDA_DIR / "package.zip") -> Path:
    if not BUILD_DIR.exists():
        print(
            f"Warning: {BUILD_DIR} doesn't exist — packaging application code "
            f"only, with NO dependencies (fastapi/mangum/pydantic). The zip "
            f"will not run on Lambda until dependencies are installed into "
            f"{BUILD_DIR} first. See this script's module docstring."
        )

    staging = LAMBDA_DIR / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # Dependencies first (if present), so application code below can
    # overwrite/coexist without needing any special ordering logic.
    if BUILD_DIR.exists():
        for item in BUILD_DIR.iterdir():
            dest = staging / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy(item, dest)

    # Application code
    shutil.copy(LAMBDA_DIR / "handler.py", staging / "handler.py")

    mcp_dest = staging / "mcp_servers"
    mcp_dest.mkdir()
    for fname in MCP_SERVER_FILES:
        shutil.copy(ROOT / "mcp_servers" / fname, mcp_dest / fname)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging))

    shutil.rmtree(staging)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Packaged Lambda deployment -> {output_path} ({size_mb:.2f} MB)")
    if size_mb > 50:
        print(
            "Note: zip exceeds Lambda's 50MB direct-upload limit — "
            "upload to S3 and reference it in deploy.py instead."
        )
    return output_path


if __name__ == "__main__":
    build_package()
