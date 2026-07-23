# Install

This document covers the full environment setup. For a quick install
on a machine that already has CUDA 12.4 + PyTorch 2.6 + Python 3.11,
`pip install -r requirements.txt` from the repo root is enough.

## Requirements

- **Python**: 3.11.x (strict)
- **OS**: Linux. We have not tested macOS or Windows.
- **GPU**: CUDA-capable NVIDIA GPU. The LoRA + NLI inference paths
  assume CUDA. The development and reference numbers in
  [REPRODUCE.md](REPRODUCE.md) were produced on an NVIDIA RTX 3090
  (24 GB VRAM, Ampere, compute capability 8.6).
- **CUDA**: 12.4 (newer minor versions in the 12.x series should also
  work).
- **Disk**: ~24 GB free for the Llama-3.2-3B-Instruct weights, two LoRA
  adapters, and the RoBERTa-MNLI cache.

## Recommended setup: conda env

We use a conda env that bundles a minimal CUDA toolkit so we don't
depend on whatever the system has installed.

```bash
conda create -n xdomain-ser python=3.11 -y
conda activate xdomain-ser

# Minimal CUDA toolkit for Flash Attention compilation.
conda install -y -c nvidia cuda-nvcc=12.4 cuda-cudart-dev=12.4

# Required env vars for nvcc + headers. Add to ~/.bashrc to persist.
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH

# Then install the dependencies.
git clone https://github.com/Vrindiesel/xdomain-ser.git
cd xdomain-ser
pip install -r requirements.txt
```

The repo is used in place: run everything from the repo root. There is
no installable package or console script in v1.0.0 — the CLI runs as
`python -m xdomain_ser.cli`.

## Flash Attention 2

The extraction and ranker LoRA inference paths request Flash Attention
2 via `attn_implementation="flash_attention_2"`. The transformers
loader will degrade gracefully to the eager backend if FA2 isn't
installed; setting `--no_flash_attn` in the affected scripts forces the
eager path explicitly.

To install FA2 from PyPI (works on glibc ≥ 2.32, i.e. Ubuntu 22.04+):

```bash
pip install flash-attn==2.5.9 --no-build-isolation
```

If your system has **glibc 2.31** (Ubuntu 20.04), the prebuilt wheels
fail with `GLIBC_2.32 not found`. Build from source:

```bash
git clone --branch v2.5.9 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install . --no-build-isolation
```

The from-source build takes ~30 minutes on a single GPU host. Flash
Attention 2.5.9 was the version validated during paper development;
later 2.x releases should also work on Ampere.

## HuggingFace authentication

The base model `meta-llama/Llama-3.2-3B-Instruct` is gated. Authenticate
once before running training, inference, or model downloads:

```bash
hf auth login                # interactive, paste a token (huggingface_hub >= 1.0)
# or for older huggingface_hub installs:
huggingface-cli login
# or non-interactive:
export HF_TOKEN=hf_xxxx
```

The two LoRA adapters we publish
(`DavanHarrison/xdomain-ser-extractor` and
`DavanHarrison/xdomain-ser-ranker`) are public and do not require
authentication, but pulling them via `transformers` triggers a base-model
load as well, so the same auth is needed.

## Optional requirement sets

`requirements.txt` pulls the core inference + evaluation stack. Four
optional sets add features:

```bash
pip install -r requirements-pbl.txt           # OpenAI Chat Completions PBL extractor
pip install -r requirements-experimental.txt  # XGBoost routing (phase-0 follow-up)
pip install -r requirements-dataprep.txt      # dataset reconstruction (data_prep)
pip install -r requirements-dev.txt           # pytest, ruff, jupyter, matplotlib
```

The PBL set is required for `xdomain_ser.extraction.pbl` (the
GPT-4o-based alternative extraction path) and the related
`scripts/reproduce_table3.sh`. The experimental set is required for
`xdomain_ser.routing.experimental.xgboost_routing` and
`scripts/reproduce_phase0.sh`. Flash Attention is installed directly
(see the section above), not via a requirements file.

## Downloading the LoRA adapters

The two adapters are pulled from the HuggingFace Hub into
`models/extractor/` and `models/ranker/` by a helper script:

```bash
bash scripts/download_models.sh
```

Re-run with `DOWNLOAD_MODELS_FORCE=1 bash scripts/download_models.sh`
to refresh.

## Verification

```bash
# 1. The package imports (run from the repo root).
python -c "import xdomain_ser; print(xdomain_ser.__version__)"

# 2. The CLI runs.
python -m xdomain_ser.cli --help

# 3. The core SER tests pass.
pytest tests/test_ser.py

# 4. The headline reproduction script runs and matches.
bash scripts/reproduce_table5.sh
```

The last command takes about 3–5 minutes on RTX 3090 and prints
`PASS: within tolerance.` if the reproduced LR-Routing All-acc lands
within ±0.01 of the paper headline (0.868).

## Troubleshooting

**`transformers` complains about `bitsandbytes`.** Install via pip
(not conda) — `pip install bitsandbytes==0.49.2`. This is the version
pinned in `requirements.txt`.

**`pip` errors with `[Errno 18] Cross-device link`.** Unset the pip
cache: `unset PIP_CACHE_DIR`. This bites us when the cache is on a
different mountpoint from the venv.

**Out of GPU memory.** The Llama-3.2-3B-Instruct base model plus a
LoRA adapter takes ~7 GB in 4-bit. The RoBERTa-MNLI baseline takes
another ~1.5 GB. On a 24 GB card all paths fit comfortably with batch
sizes up to 32; on 16 GB cards drop the inference batch to 4 or 8
(`--batch_size 4` in `xdomain_ser.extraction.inference`).

**Flash Attention `Imports` error from `wandb`.** A wandb / transformers
version mismatch — we don't use wandb anywhere in the release, but
some transitive imports trigger it. Set `WANDB_DISABLED=true` or
uninstall wandb.

**`np.asfarray` missing in numpy ≥ 2.x.** The IR ranking metrics
module (`xdomain_ser.core.rank_metrics.dcg_at_k`) uses `np.asfarray`
which was removed in NumPy 2.0. The pinned `numpy<2` in
`requirements.txt` should keep you safe, but if you upgrade NumPy,
swap to `np.asarray(r, dtype=float)`.
