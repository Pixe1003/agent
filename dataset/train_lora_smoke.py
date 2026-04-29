"""Small LoRA smoke-test entrypoint placeholder.

This script intentionally avoids importing heavy training dependencies at module
import time. Install a training stack such as LLaMA-Factory or axolotl before
turning the generated JSONL into a real run.
"""

from pathlib import Path


def check_dataset(path: str = "dataset/cloud-sched-sft-v1.jsonl") -> int:
    dataset = Path(path)
    if not dataset.exists():
        raise FileNotFoundError(path)
    return sum(1 for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    print(f"dataset samples: {check_dataset()}")

