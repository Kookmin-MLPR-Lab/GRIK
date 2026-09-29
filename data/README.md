# LongBench v1 data

Run the helper below from the repo root to fetch the LongBench-v1 task files
into `data/LongBench/` (16 tasks × 200–500 jsonl rows ≈ 360 MB):

```bash
mkdir -p data/LongBench && cd data/LongBench
curl -L -o data.zip https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
unzip data.zip && rm data.zip
```

Optional (RULER, only if reproducing the long-context appendix table):
see `scripts/run_ruler.py --help` for prep instructions; raw RULER prompts
are produced by NVIDIA's harness at
<https://github.com/NVIDIA/RULER>.

Model weights are not bundled; download e.g.:

```bash
huggingface-cli download unsloth/Meta-Llama-3-8B-Instruct       --local-dir .models/llama-3-8b
huggingface-cli download unsloth/Meta-Llama-3.1-8B-Instruct     --local-dir .models/llama-3.1-8b
huggingface-cli download unsloth/Qwen2.5-7B-Instruct            --local-dir .models/qwen2.5-7b
```
