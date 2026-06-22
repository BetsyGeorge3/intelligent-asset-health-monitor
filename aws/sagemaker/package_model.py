"""
Package the trained BiLSTM model into the model.tar.gz structure
SageMaker's PyTorch inference containers expect:

    model.tar.gz
    ├── bilstm_weights.pt
    ├── model_metadata.json
    └── code/
        ├── inference.py     (SageMaker entry point — model_fn etc.)
        ├── bilstm.py         (model architecture)
        └── dataset.py        (Normalizer + SENSOR_COLS)

Usage:
    cd asset_health_monitor
    python aws/sagemaker/package_model.py
    # → produces aws/sagemaker/model.tar.gz
"""

import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SAGEMAKER_DIR = Path(__file__).parent
SAVED_MODEL_DIR = ROOT / "models" / "saved"

WEIGHTS_FILE = SAVED_MODEL_DIR / "bilstm_weights.pt"
METADATA_FILE = SAVED_MODEL_DIR / "model_metadata.json"


def build_package(output_path: Path = SAGEMAKER_DIR / "model.tar.gz") -> Path:
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"No trained model found at {WEIGHTS_FILE}. Run `python models/train.py` first."
        )

    staging = SAGEMAKER_DIR / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    code_dir = staging / "code"
    code_dir.mkdir()

    # Model artifacts at the tarball root (SageMaker convention)
    shutil.copy(WEIGHTS_FILE, staging / "bilstm_weights.pt")
    shutil.copy(METADATA_FILE, staging / "model_metadata.json")

    # Inference code + the two modules it imports, under code/
    shutil.copy(SAGEMAKER_DIR / "inference.py", code_dir / "inference.py")
    shutil.copy(ROOT / "models" / "bilstm.py", code_dir / "bilstm.py")
    shutil.copy(ROOT / "models" / "dataset.py", code_dir / "dataset.py")

    with tarfile.open(output_path, "w:gz") as tar:
        for item in staging.iterdir():
            tar.add(item, arcname=item.name)

    shutil.rmtree(staging)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Packaged model -> {output_path} ({size_mb:.2f} MB)")
    return output_path


if __name__ == "__main__":
    build_package()
