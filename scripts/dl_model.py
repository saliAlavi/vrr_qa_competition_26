"""Pre-download a HF model snapshot to the scratch cache (login node, reliable net).
Usage: HF_HOME=/fs/scratch/PAS2301/alialavi/cache python scripts/dl_model.py <repo_id>
Avoids the in-job vLLM download hang on multi-GPU nodes.
"""
import sys, time
from huggingface_hub import snapshot_download

mid = sys.argv[1]
for attempt in range(6):
    try:
        p = snapshot_download(mid, allow_patterns=[
            "*.safetensors", "*.json", "*.txt", "*.model", "merges.txt",
            "vocab.json", "*.py", "tokenizer*", "*.jinja"])
        print(f"DONE {mid} -> {p}", flush=True)
        break
    except Exception as e:
        print(f"retry {attempt} {mid}: {e}", flush=True)
        time.sleep(10)
