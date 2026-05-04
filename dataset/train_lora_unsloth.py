"""Unsloth + Qwen2.5-1.5B LoRA SFT 训练脚本。

Colab 用法（推荐 T4 免费版即可）：
    1. 把整个项目 (或至少 dataset/cloud-sched-sft-v2.jsonl) 上传到 Colab
    2. 一个 cell 安装 Unsloth：
       !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
       !pip install --no-deps trl<0.9.0 peft accelerate bitsandbytes
    3. 一个 cell 跑：!python dataset/train_lora_unsloth.py

本地 GPU 用法 (RTX 3060+ / 8GB+ VRAM)：
    pip install unsloth trl peft accelerate bitsandbytes
    python dataset/train_lora_unsloth.py

输出：
    dataset/qwen25-1p5b-sched-lora/                  (LoRA adapter, ~30 MB)
    dataset/qwen25-1p5b-sched-merged-q4.gguf         (合并 + 4bit 量化, 推理用, ~1 GB)

时长预估：T4 上 12k 样本 × 3 epoch ≈ 35-50 min
"""
from __future__ import annotations

import json
from pathlib import Path

# 这些 import 只在跑训练时需要，平时模块导入不会失败
def _check_env():
    try:
        import unsloth  # noqa: F401
        import trl       # noqa: F401
    except ImportError:
        raise SystemExit(
            "Missing dependencies. Install:\n"
            '  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'
            "  pip install --no-deps trl<0.9.0 peft accelerate bitsandbytes"
        )


# ============================================================================
# 配置
# ============================================================================

DATASET_PATH = Path("dataset/cloud-sched-sft-v2.jsonl")
BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
LORA_OUTPUT_DIR = Path("dataset/qwen25-1p5b-sched-lora")
GGUF_OUTPUT = Path("dataset/qwen25-1p5b-sched-merged-q4.gguf")

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

MAX_SEQ_LENGTH = 2048
TRAIN_EPOCHS = 3
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 4         # effective batch = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
SEED = 42


def main() -> None:
    _check_env()

    if not DATASET_PATH.exists():
        raise SystemExit(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run first:  python -m dataset.build_sft_dataset --v2 --max-samples 12000"
        )

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"[1/5] Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,            # auto = bf16 on Ampere+, fp16 on T4
        load_in_4bit=True,
    )

    print(f"[2/5] Wrapping with LoRA (r={LORA_R}, alpha={LORA_ALPHA})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    print(f"[3/5] Loading dataset: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
    dataset = dataset.shuffle(seed=SEED)
    print(f"    samples: {len(dataset)}")
    if len(dataset) < 500:
        print("    WARNING: <500 samples, model will likely overfit. "
              "Run benchmark.runner with more seeds first.")

    # SFTTrainer 直接吃 'text' 字段（v2 格式已经是预渲染的 ChatML）
    print(f"[4/5] Training: {TRAIN_EPOCHS} epoch × bs {PER_DEVICE_BATCH} × accum {GRAD_ACCUM}")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            output_dir=str(LORA_OUTPUT_DIR),
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER,
            optim="adamw_8bit",
            logging_steps=20,
            save_strategy="epoch",
            save_total_limit=1,
            seed=SEED,
            bf16=True,
            fp16=False,
            report_to="none",
        ),
    )
    train_stats = trainer.train()
    print(f"    train loss: {train_stats.training_loss:.4f}")

    print(f"[5/5] Saving LoRA adapter to {LORA_OUTPUT_DIR}")
    model.save_pretrained(str(LORA_OUTPUT_DIR))
    tokenizer.save_pretrained(str(LORA_OUTPUT_DIR))

    # 顺手存一份 GGUF q4 (推理用，llama-cpp-python 直接吃)
    print(f"      Exporting merged GGUF q4 to {GGUF_OUTPUT}")
    try:
        model.save_pretrained_gguf(
            str(GGUF_OUTPUT.parent),
            tokenizer,
            quantization_method="q4_k_m",
        )
        print("      GGUF saved.")
    except Exception as e:
        print(f"      GGUF export skipped ({type(e).__name__}: {e}). "
              "在 Colab 里有时需要先 !apt install build-essential。")

    print()
    print("Done. Next:")
    print("  1) cd 回项目根，把 LoRA / GGUF 拷回本地")
    print("  2) python -m benchmark.runner   # 会自动跑 AI-sft-1.5b 算法分支")


if __name__ == "__main__":
    main()
