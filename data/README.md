# Data Sources

## POJ-104 (Microsoft CodeXGLUE clone-detection task)
- Microsoft CodeXGLUE README: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-POJ-104
- Hugging Face mirror used by the support material: https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_poj104
- Mirror endpoint that worked at preparation time: https://hf-mirror.com

## Layout
- `poj104/train.jsonl`, `valid.jsonl`, `test.jsonl` — exported splits from the Hugging Face mirror, with the same row counts (32500 / 8500 / 12000) and SHA256 fingerprints recorded in the original `result_summary.json`.

The rest of the support material (preprocessed token / AST-proxy / normalized-id text) is regenerated deterministically from these JSONL files when `run_reproduction.py` is launched, so only the raw splits are shipped here.
