# Instrumentation Loop Engineering AI — Unsloth LoRA Starter Kit

This kit is a starting point for fine-tuning an open-weight language model to assist with
instrumentation loop engineering for oil & gas, refinery, petrochemical, and steel-plant projects.

## Important engineering boundary

This is an AI training starter kit, not an engineering design package or approval system.
Never use generated loop details as construction, commissioning, or safety instructions without
review by a qualified instrumentation/control engineer against the approved project documents,
applicable codes/standards, hazardous-area classification, vendor manuals, and site procedures.
The sample data is illustrative and must not be treated as a real plant design.

## What is included

- `training/train_lora.py`: Unsloth-accelerated QLoRA supervised fine-tuning.
- `training/infer.py`: FastAPI inference API that also serves the built React interface.
- `frontend/`: React/Vite prompt interface for inference.
- `dataset/train.jsonl`: illustrative chat-format training examples.
- `dataset/validation.jsonl`: illustrative validation examples.
- `requirements.txt`: Python packages for a Linux/WSL2 NVIDIA environment.
- `.env.example`: configurable model and training settings.
- `docs/DATA_PREPARATION.md`: how to build a project-specific dataset safely.
- `docs/MODEL_CHOICES.md`: choosing a model and understanding Groq deployment limits.
- `docs/ENGINEERING_SCOPE.md`: suggested subject areas and verification checklist.

## Recommended first experiment

Default base model: `Qwen/Qwen2.5-7B-Instruct`.

Why: it is a practical, open-weight starting point supported by the Unsloth fine-tuning workflow.
It is not the same model as `qwen/qwen3.8-27b` or `openai/gpt-oss-20b` in your Groq configuration.
Model availability, licenses, hardware needs, and library compatibility can change; check the model
card before use.

For an initial test, use a CUDA-enabled Linux machine or Ubuntu on WSL2. A 7B QLoRA run commonly
needs a capable NVIDIA GPU; exact VRAM depends on sequence length, batch size, quantization,
checkpointing, and library versions. If you do not have a suitable GPU, use a rented GPU VM.

## 1. Create the environment (Ubuntu / WSL2)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
mkdir instrumentation-loop-ai
cd instrumentation-loop-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install Unsloth and its dependencies using the official installation instructions for your
OS, Python version, and CUDA/PyTorch combination:
https://unsloth.ai/docs/get-started/install/pip-install
On Linux/WSL2, follow the current pip or uv instructions and select a compatible PyTorch backend
before installing this project's requirements.

Then install the kit dependencies:

```bash
pip install -r requirements.txt
```

Unsloth's supported platform and CUDA/PyTorch combinations change over time. Check its current
installation guide instead of assuming native Windows or a particular CUDA version is supported.

## 2. Get the base model

The training script downloads the model from Hugging Face automatically when run. You can also
pre-download it:

```bash
pip install -U huggingface_hub
hf download Qwen/Qwen2.5-7B-Instruct
```

If the repository requires accepting terms or authentication, follow its model-card instructions
and run `hf auth login`.

## 3. Prepare your data

Review `docs/DATA_PREPARATION.md`. Replace the illustrative examples with approved, de-identified,
project-authorized examples. Do not upload confidential plant drawings, client information,
credentials, or controlled documents to a third-party service without authorization.

The JSONL files use one JSON object per line, with a `messages` array containing `user` and
`assistant` messages.

## 4. Configure training

Copy `.env.example` to `.env` and set the values, or export environment variables directly.
The script reads environment variables; it does not automatically parse `.env`. To load `.env`,
install `python-dotenv` and uncomment the optional lines in the script, or export the variables
in your shell.

Example:

```bash
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export TRAIN_FILE="dataset/train.jsonl"
export VALIDATION_FILE="dataset/validation.jsonl"
export OUTPUT_DIR="outputs/instrumentation-qwen-lora"
export MAX_SEQ_LENGTH="2048"
export EPOCHS="2"
export LEARNING_RATE="2e-4"
```

## 5. Run fine-tuning

```bash
python training/train_lora.py
```

The script uses Unsloth's 4-bit model loading, LoRA setup, and gradient-checkpointing path with
TRL's supervised fine-tuning trainer. It saves the LoRA adapter and tokenizer in the output
directory; it does not replace or rewrite the original base model weights. Training and inference
must use compatible Unsloth, Transformers, and TRL dependencies.

## 6. Run the inference UI

Set the adapter directory and, optionally, a different base model. The adapter path can be
absolute or relative to the project root.

```bash
export ADAPTER_PATH="outputs/instrumentation-qwen-lora"
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

Build the React interface and start the FastAPI server:

```bash
cd frontend
npm install
npm run build
cd ..
uvicorn training.infer:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. The model loads on the first inference request and is kept
in memory for subsequent requests. For frontend development, run `npm run dev` in `frontend`
and start Uvicorn separately; Vite proxies `/api` requests to the FastAPI server.

## 7. Evaluate before using

Create a held-out test set that was not used in training. Have qualified engineers score:
- correct identification of AI/AO/DI/DO and signal direction;
- field-to-JB-to-marshalling-to-system path consistency;
- distinction between sourced facts and assumptions;
- correct handling of missing project details (ask, do not invent);
- hazardous-area / intrinsic-safety caveats;
- cable, shield, grounding, and power checks where relevant;
- refusal to claim compliance without the governing project standard and evidence.

Do not evaluate only by whether the answer sounds professional. Check every technical claim.

## Groq note

Groq is an inference provider, not a general-purpose PEFT training environment for arbitrary
models. Training an adapter locally or on a cloud GPU does not mean that adapter can be uploaded
to Groq. Confirm Groq's current supported model and LoRA deployment options before planning
deployment. A Qwen2.5 adapter cannot simply be attached to a different Groq model family.

## Useful official references

- Unsloth installation: https://unsloth.ai/docs/get-started/install/pip-install
- Unsloth fine-tuning guide: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
- Hugging Face PEFT: https://huggingface.co/docs/peft
- Hugging Face TRL (trainer API used by Unsloth): https://huggingface.co/docs/trl
- Transformers: https://huggingface.co/docs/transformers
- Qwen2.5-7B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- PyTorch installation: https://pytorch.org/get-started/locally/
- Groq documentation: https://console.groq.com/docs

## Troubleshooting

- CUDA unavailable: install the matching PyTorch CUDA build and NVIDIA driver.
- Out of memory: lower `MAX_SEQ_LENGTH`, `BATCH_SIZE`, or `GRAD_ACCUM`; use a smaller model/GPU.
- Dataset parse error: validate every line as JSON and ensure each row has a `messages` list.
- Unsloth / TRL API mismatch: reinstall using Unsloth's current installation instructions and
  consult the Unsloth and TRL trainer documentation.
- Model access denied: check the model card, license, access approval, and Hugging Face login.
