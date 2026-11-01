# Model choice and deployment notes

## Starter recommendation

Use `Qwen/Qwen2.5-7B-Instruct` as the example base model in this kit. It is selected for an
accessible Hugging Face model that can be used with the Unsloth workflow in this kit. Confirm its
current model card, license and compatibility before use.

## Your original Groq model IDs

- `openai/gpt-oss-20b`
- `openai/gpt-oss-120b`
- `qwen/qwen3.8-27b`

These are inference model identifiers in your Groq configuration. They are not interchangeable
with Hugging Face repository IDs. Do not assume a Groq model endpoint provides downloadable
weights or that a LoRA adapter trained for one model family can be applied to another.

## GPT-OSS

GPT-OSS has its own model repository and training guidance. If you specifically want to fine-tune
`openai/gpt-oss-20b`, follow the official GPT-OSS recipe and verify current support for the exact
training method, library versions, hardware, and adapter format. Do not use the Qwen training
script in this kit unchanged for GPT-OSS.

Official starting points:
- https://github.com/openai/gpt-oss
- https://developers.openai.com/cookbook/topic/gpt-oss

## Groq deployment

Training and inference are separate. You can train on your own GPU or a cloud GPU, but uploading
an adapter to Groq is possible only if Groq currently supports that exact base model, adapter
format, and account tier. Check current Groq documentation before choosing a deployment path:
https://console.groq.com/docs

If adapter hosting is not supported for your selected model, serve the base model plus adapter
on your own compatible inference infrastructure, or use a supported hosted model/API. Keep the
model architecture consistent between training and inference.
