# GRIK: Grouped Relevance-Integrated Key Cache Pruning

> Reference implementation for the paper
> *GRIK: Grouped Relevance-Integrated Key Cache Pruning for Grouped-Query
> Attention LLMs.*

GRIK is a **training- and calibration-free** channel-pruning method for the
key (K) cache of grouped-query attention (GQA) LLMs. It assigns a single
shared mask per KV head so that channel sparsity translates one-to-one into
realized cache savings — a property that ThinK (per-Q-head pruning) breaks
under GQA storage and SparK (unstructured per-token pruning) does not
provide at all.

The repo is the minimal subset of the development codebase needed to
reproduce the paper's main LongBench-v1 table and the long-context RULER
results.

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── data/                   # download-on-demand; not bundled
│   └── README.md
├── paper/
│   ├── main.tex            # paper source
│   ├── figs/               # figures (motivation, scorecard, pareto)
│   └── tables/
│       └── main_table_longbench.tex
├── scripts/
│   ├── setup.sh            # one-shot env + data bootstrap
│   ├── run_longbench.py    # main eval harness (16 LongBench-v1 tasks)
│   ├── run_ruler.py        # long-context RULER eval (optional)
│   ├── eval.py             # offline metric computation
│   └── metrics.py          # F1 / ROUGE / classification scoring
└── src/
    ├── monkeypatch.py      # patches transformers' LlamaAttention.forward
    ├── llama_model.py      # GRIK-aware attention forward + cache prep
    ├── kv_pruning_utils.py # iterative scorer (GRIK), SparK port, H2O state
    └── cache_utils.py      # eviction-aware cache datastructures
```

## Method (one-paragraph summary)

For each KV head $h_{kv}$, GRIK aggregates the influence of the $g$ query
heads in its group through their output-projection contributions
$\alpha_h = \lVert W_O[:, h \cdot d : (h{+}1) \cdot d]\rVert_2$, then ranks
channels by a score that combines the static K-projection row norm
(Wanda-style, $\lVert k_{\text{proj}}\rVert$) with the dynamic energy
$Q^2 \cdot K^2 \cdot \alpha_h$. The mask is refined for $N{=}32$
progressive-sparsity steps from full retention down to $\lfloor r d \rfloor$
channels, and the resulting GQA-shared mask gates a dense pruned cache of
shape $\mathbb{R}^{H_{kv} \times T \times \lfloor r d \rfloor}$. No
calibration data, no training, no per-prompt parameters.

The full algorithm is in `src/kv_pruning_utils.py::key_pruner_iterative`.

## Headline results

### Main table — LongBench-v1 16-task average

Three GQA backbones, four KV budgets *B*, retention ratios *r* ∈ {0.4, 0.6}
(H2O eviction). **Bold** marks the best non-Full-KV cell per (model, *B*, *r*).
Rightmost column: asymptotic K-cache ratio vs. Full KV (GRIK = *r*,
SparK = 1.00×, ThinK = *g* · *r*).

| Model | Method | B=128 | B=512 | B=1024 | B=2048 | K-cache |
|---|---|---:|---:|---:|---:|---:|
| **LLaMA-3-8B-Inst.** | Full KV (reference)<sup>\*</sup> | 35.38 | 37.23 | 38.70 | 39.59 | 1.00× |
|  | ThinK (r=0.4) | 35.63 | 37.39 | 39.00 | 40.05 | 1.60× |
|  | SparK (r=0.4) | 32.73 | 33.92 | 34.68 | 34.83 | 1.00× |
|  | **GRIK (r=0.4)** | **35.92** | **38.63** | **40.12** | **41.30** | **0.40×** |
|  | ThinK (r=0.6) | 35.06 | 36.44 | 37.58 | 38.51 | 2.40× |
|  | SparK (r=0.6) | 27.97 | 26.62 | 24.11 | 21.92 | 1.00× |
|  | **GRIK (r=0.6)** | **35.00** | **37.45** | **38.45** | **39.24** | **0.60×** |
| **LLaMA-3.1-8B-Inst.** | Full KV (reference) | 39.87 | 43.38 | 44.88 | 45.73 | 1.00× |
|  | ThinK (r=0.4) | 39.22 | 41.56 | 42.79 | 44.30 | 1.60× |
|  | SparK (r=0.4) | 38.22 | 39.94 | 39.94 | 38.47 | 1.00× |
|  | **GRIK (r=0.4)** | **39.60** | **43.17** | **44.63** | **45.46** | **0.40×** |
|  | ThinK (r=0.6) | 39.00 | **40.89** | **42.38** | 43.65 | 2.40× |
|  | SparK (r=0.6) | 34.06 | 29.86 | 25.12 | 20.65 | 1.00× |
|  | **GRIK (r=0.6)** | **39.31** | **42.07** | **43.54** | **44.62** | **0.60×** |
| **Qwen2.5-7B-Inst.** | Full KV (reference) | 39.58 | 42.35 | 43.79 | 44.89 | 1.00× |
|  | ThinK (r=0.4) | **39.41** | 42.16 | 43.68 | 45.13 | 2.80× |
|  | SparK (r=0.4) | 10.73 | 11.41 | 11.49 | 11.28 | 1.00× |
|  | **GRIK (r=0.4)** | 38.88 | **42.53** | **43.98** | **45.29** | **0.40×** |
|  | ThinK (r=0.6) | **39.40** | 41.93 | **43.64** | **44.92** | 4.20× |
|  | SparK (r=0.6) | 10.55 | 10.93 | 10.98 | 10.86 | 1.00× |
|  | **GRIK (r=0.6)** | 38.27 | **42.05** | 43.49 | 44.73 | **0.60×** |

<sup>\*</sup> LLaMA-3-8B Full KV / ThinK rows cited from ThinK [Xu et al., ICLR 2025], Table 2.

GRIK is the strongest non-Full-KV method on **19 / 24** cells, within 0.6
points of the leader on 4 of the remaining 5, within 1.13 on the last.
On LLaMA-3.1-8B at *r*=0.4 GRIK trails Full KV by at most 0.27 points at
every budget while ThinK loses 0.65–2.09 and SparK 1.65–7.26.

### Long-context retrieval — RULER on LLaMA-3.1-8B-Instruct

Channel-only regime (*r*=0.7, KV budget 200K so H2O eviction never fires).

| Method | 4K | 8K | 16K | 32K | Avg. |
|---|---:|---:|---:|---:|---:|
| LLaMA-3.1-8B<sup>†</sup> | 95.10 | 93.10 | 90.20 | 86.00 | 91.10 |
| ThinK<sup>†</sup> | 58.30 | 39.20 | 37.50 | 36.40 | 42.85 |
| SparK | 6.17 | 3.80 | 3.32 | 2.58 | 3.97 |
| **GRIK (Ours)** | **61.52** | **60.00** | **56.70** | **54.24** | **58.12** |

<sup>†</sup> Cited at the matched setting; Avg. recomputed over the four
lengths shown.

SparK averages below 7 at every length; all eight needle-in-a-haystack
subtasks score 0% (per-token recovery fills dropped channels with noise,
destroying the precise key signature retrieval requires). GRIK retains
54–62 over the same range — condition C2 (token consistency) is
functionally required for retrieval-style attention.

### Score-form ablation — LLaMA-3-8B-Instruct, LongBench-AVG

Holding the storage layout fixed (single H<sub>kv</sub>×*k* K cache),
GRIK adds three score-form ingredients over the uniform-mean baseline
(ThinK-GQA-mean). The gap widens with budget.

| Method | B=128 | B=512 | B=1024 | B=2048 |
|---|---:|---:|---:|---:|
| **r = 0.4** | | | | |
| ThinK-GQA-mean | 35.45 | 36.96 | 38.11 | 39.48 |
| **GRIK (Ours)** | **35.92** | **38.63** | **40.12** | **41.30** |
| Δ | +0.47 | +1.67 | +2.01 | +1.82 |
| **r = 0.6** | | | | |
| ThinK-GQA-mean | 34.91 | 36.30 | 37.40 | 38.30 |
| **GRIK (Ours)** | **35.00** | **37.45** | **38.45** | **39.24** |
| Δ | +0.09 | +1.15 | +1.05 | +0.94 |

### K-cache footprint vs. Full KV

| Method | B=128 | B=512 | B=1024 | B=2048 |
|---|---:|---:|---:|---:|
| **r = 0.4** | | | | |
| ThinK | 3.41× | 3.26× | 3.24× | 3.23× |
| SparK | 1.00× | 1.00× | 1.00× | 1.00× |
| **GRIK** | **0.85×** | **0.82×** | **0.81×** | **0.81×** |
| **r = 0.6** | | | | |
| ThinK | 3.12× | 2.90× | 2.86× | 2.84× |
| SparK | 1.00× | 1.00× | 1.00× | 1.00× |
| **GRIK** | **0.78×** | **0.72×** | **0.71×** | **0.71×** |

In a 14 GiB KV pool typical of an 8B-class model on a 48 GiB GPU, GRIK
fits roughly 1.25× as many context tokens as Full KV, while ThinK fits
roughly 0.31×. GRIK composes with FP8 KV quantization for 0.40× the
bf16 Full-KV baseline (Appendix).

Per-task tables, no-eviction column, and prefill latency in the paper
(`paper/main.tex`).

## Installation

```bash
git clone https://github.com/hyunjoon0208/GRIK-Grouped-Relevance-Integrated-Key-Cache-Pruning-for-Grouped-Query-Attention-LLMs.git GRIK
cd GRIK
bash scripts/setup.sh                # creates .venv, installs deps, fetches LongBench data
source .venv/bin/activate
```

Tested with `transformers ∈ [4.40, 4.45]`, `torch ≥ 2.1`, single A100/H100
or RTX PRO 6000.

Download the model checkpoints separately (paths configurable):

```bash
huggingface-cli download unsloth/Meta-Llama-3-8B-Instruct    --local-dir .models/llama-3-8b
huggingface-cli download unsloth/Meta-Llama-3.1-8B-Instruct  --local-dir .models/llama-3.1-8b
huggingface-cli download unsloth/Qwen2.5-7B-Instruct         --local-dir .models/qwen2.5-7b
```

## Reproducing GRIK on LongBench

This build registers a single method, **`grik`** — the canonical paper
configuration (32 iterations + Wanda K-norm prior + Q²·K²·α dynamic).
A single LongBench-v1 run takes ~30–60 min on a single A100/H100 at
*B*=128, ~1.5–4 h at *B*=2048 depending on backbone and ratio.

```bash
python scripts/run_longbench.py \
    --model_path .models/llama-3-8b \
    --method grik \
    --max_capacity_prompts 1024 \
    --pruning_ratio 0.4 \
    --recent_size 32 \
    --save_dir results/grik_kv1024_r04
```

Then score:

```bash
python scripts/eval.py --pred_dir results/grik_kv1024_r04
```

The hyperparameters that define the paper's main-table sweep are simply
`--max_capacity_prompts ∈ {128, 512, 1024, 2048, 200000}` (the last
disables H2O eviction) and `--pruning_ratio ∈ {0.4, 0.6}`. ThinK and
SparK baselines (cited in the headline tables above) live in the
authors' original repos; they are intentionally not bundled here so that
this repository contains only the proposed method.

## Reproducing the RULER table (long-context, optional)

RULER (NVIDIA) measures retrieval and synthetic reasoning at 4K – 128K. To
reproduce the LLaMA-3.1 RULER cells in the paper:

```bash
# 1. Prepare RULER data (one-time, see https://github.com/NVIDIA/RULER)
# 2. Run a single (method, ctx) cell:
python scripts/run_ruler.py \
    --model_path .models/llama-3.1-8b \
    --method grik \
    --max_capacity_prompts 200000 \
    --pruning_ratio 0.7 --recent_size 32 \
    --data_dir <RULER prepared dir>/32768/data \
    --save_dir results/ruler_grik_32k \
    --task niah_single_1
```

KV=200K means the budget is larger than the longest sequence, so H2O
eviction never fires — this isolates the channel-pruning quality.

## Citation

```bibtex
@article{grik2026,
  title  = {GRIK: Grouped Relevance-Integrated Key Cache Pruning for Grouped-Query Attention LLMs},
  author = {Anonymous},
  year   = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
