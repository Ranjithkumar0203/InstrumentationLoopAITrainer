import os
from pathlib import Path

from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


def env(name, default):
    return os.environ.get(name, str(default))


BASE_MODEL = env("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
TRAIN_FILE = env("TRAIN_FILE", "dataset/train.jsonl")
VALIDATION_FILE = env("VALIDATION_FILE", "dataset/validation.jsonl")
OUTPUT_DIR = env("OUTPUT_DIR", "outputs/instrumentation-qwen-lora")
MAX_SEQ_LENGTH = int(env("MAX_SEQ_LENGTH", 2048))
EPOCHS = float(env("EPOCHS", 2))
LEARNING_RATE = float(env("LEARNING_RATE", 2e-4))
BATCH_SIZE = int(env("BATCH_SIZE", 1))
GRAD_ACCUM = int(env("GRAD_ACCUM", 8))
LORA_R = int(env("LORA_R", 16))
LORA_ALPHA = int(env("LORA_ALPHA", 32))
LORA_DROPOUT = float(env("LORA_DROPOUT", 0.05))

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA GPU not detected. This QLoRA starter expects a supported NVIDIA CUDA "
        "environment. Use WSL2/Linux or a cloud GPU and install the matching PyTorch build."
    )

for path in (TRAIN_FILE, VALIDATION_FILE):
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

# Unsloth QLoRA uses 4-bit base weights; prefer bfloat16 when supported.
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=compute_dtype,
    load_in_4bit=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
)
model.config.use_cache = False

dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE, "validation": VALIDATION_FILE},
)

def format_chat(example):
    # Apply the model's own chat template; messages must use supported role names.
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = dataset.map(format_chat)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    logging_steps=int(env("LOGGING_STEPS", 10)),
    save_steps=int(env("SAVE_STEPS", 100)),
    eval_strategy="steps",
    eval_steps=int(env("SAVE_STEPS", 100)),
    save_total_limit=2,
    bf16=(compute_dtype == torch.bfloat16),
    fp16=(compute_dtype == torch.float16),
    optim="paged_adamw_8bit",
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to: {OUTPUT_DIR}")
