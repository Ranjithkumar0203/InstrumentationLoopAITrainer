# Preparing a useful instrumentation engineering dataset

## 1. Use authorized source material

Use only documents you are permitted to use for model development. Remove client names, site names,
asset identifiers, personal data, credentials, network details, and commercially sensitive material
unless your organization has explicitly approved their use. Keep an auditable source register.

## 2. Suggested dataset topics

- Instrument identification and tag conventions
- P&ID symbol/tag interpretation
- Instrument datasheet fields and units
- Analog and discrete signal paths
- DCS/PLC I/O types and channel constraints
- Junction box and marshalling termination logic
- Cable type, pair/core selection and segregation requirements
- Shield and grounding philosophy
- Control valve, positioner, solenoid and limit-switch interfaces
- HART and digital communication considerations
- Hazardous-area protection and intrinsic-safety review questions
- Loop checking and commissioning documentation
- Document cross-checking and discrepancy reporting

## 3. Use consistent examples

Each training row should teach one clear task. Include the source context, the requested output,
and a correct answer reviewed by an engineer. Include examples where the correct response is to
ask for missing information rather than guess.

## 4. Chat JSONL format

One JSON object per line:

```json
{"messages":[
  {"role":"user","content":"Your engineering question and supplied project facts"},
  {"role":"assistant","content":"A reviewed answer with assumptions and checks"}
]}
```

Do not include trailing commas. Validate the JSONL before training.

## 5. Quality controls

- Split by project or document set, not just random rows, to reduce leakage.
- Keep a held-out validation/test set.
- Include both common cases and edge cases.
- Preserve units, tag numbers, signal direction and source references.
- Mark unknown data as unknown; never fill gaps with plausible-sounding guesses.
- Have a qualified instrumentation/control engineer review all target answers.
- Track dataset version, source document revision, reviewer and approval status.

## 6. Fine-tuning versus RAG

Use PEFT to teach response format, terminology, task patterns and consistent workflow.
Use retrieval-augmented generation (RAG) to fetch current project-specific facts from controlled
documents such as approved P&IDs, datasheets, I/O lists, cable schedules and standards.
Do not expect fine-tuning to reliably memorize changing project drawings or revisions.
