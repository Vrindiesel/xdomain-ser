# xdomain-ser

[![CI](https://github.com/Vrindiesel/xdomain-ser/actions/workflows/ci.yml/badge.svg)](https://github.com/Vrindiesel/xdomain-ser/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](INSTALL.md)

A learned, cross-domain Slot Error Rate (SER) metric for evaluating
meaning-to-text (M2T) generation. Companion code, gold annotations, and
trained adapters for *Cross-Domain Semantic Fidelity Evaluation for
Meaning-to-Text Generation* (GEM @ ACL 2026).

Rule-based slot aligners need a hand-engineered evaluator per domain.
We train one extraction model on a multi-domain corpus
(Llama-3.2-3B-Instruct + LoRA), use over-generate-and-rank to pick the
best of `k` beam-search candidates, and verify slot-level coverage with
an NLI entailment model. Per-example routing between the LoRA
extraction and NLI verification reaches **0.868 All-acc** on the
multi-domain test split — beating both methods used in isolation
(0.764 LoRA-only All-acc, 0.805 NLI-only All-acc).

## Requirements

- Python 3.11 (strict)
- CUDA-capable GPU (the LoRA + NLI inference paths assume CUDA)
- ~24 GB system disk for the LoRA adapters + RoBERTa-MNLI cache

See [INSTALL.md](INSTALL.md) for the full environment (CUDA 12.4,
Flash Attention 2, glibc 2.31 build notes).

## Install

```bash
pip install -r requirements.txt
```

There is no packaged install in v1.0.0 — run everything from the repo
root, with the CLI as `python -m xdomain_ser.cli`. For the optional
GPT-4o-based PBL extractor or the phase-0 follow-up experiments:

```bash
pip install -r requirements-pbl.txt
pip install -r requirements-experimental.txt
```

## 30-second quickstart

Score a single (text, MR) pair against the E2E domain hint map:

```bash
python -m xdomain_ser.cli score \
  --text "The Vaults is a cheap restaurant near the river." \
  --gold-mr '{"name":"The Vaults","priceRange":"cheap","near":"river"}' \
  --domain e2e
```

Output (abbreviated):

```
Predicted MR: {"name":"The Vaults", "eatType":"restaurant",
               "priceRange":"cheap", "area":"riverside"}
  SER:    1.0000  (S=0  D=1  I=2  N_ref=3)
  Slot F1: 0.5714
```

The extractor recovered `name` and `priceRange` directly, read "near
the river" as `area=riverside` rather than `near=river`, and added
`eatType=restaurant` as an insertion relative to the user's partial
gold. See [USAGE.md](USAGE.md) for full worked examples and instructions
for writing a hint map for a new domain.

## Results

SER-agreement accuracy (All-acc) on the multi-domain Eval-2 test
split, published P-oracle protocol:

| Method | All-acc |
|---|---|
| LoRA extraction + ranking | 0.764 |
| NLI baseline | 0.805 |
| **Per-example routing (LR)** | **0.868** |
| XGBoost routing (phase-0 follow-up, experimental) | 0.900 |

`reproduce_table5.sh` and `reproduce_phase0.sh` validate the two gated
headlines against these numbers. The corrected per-pair deploy protocol
yields lower absolute numbers for every method — see the Eval-2
conditioning correction in [CHANGELOG.md](CHANGELOG.md) and the
artifacts under `evaluation/results/eval2_corrected/`.

## What ships

| Layer | Where |
|---|---|
| Python package | `xdomain_ser/` |
| CLI (`python -m xdomain_ser.cli`) | `xdomain_ser/cli.py` (Typer app) |
| LoRA MR-extraction adapter | HuggingFace Hub: [`DavanHarrison/xdomain-ser-extractor`](https://huggingface.co/DavanHarrison/xdomain-ser-extractor) |
| LoRA ranker adapter | HuggingFace Hub: [`DavanHarrison/xdomain-ser-ranker`](https://huggingface.co/DavanHarrison/xdomain-ser-ranker) |
| Multi-domain SER eval set | `data/multi_ser_v9/` |
| 1,000-example gold-annotated set | `evaluation/gold/` |
| Rule-based aligner baselines | `xdomain_ser/baselines/` (E2E, RNNLG, ViGGO) |
| Reproduction scripts | `scripts/reproduce_*.sh` |

The LoRA adapters sit on top of
[`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct);
the NLI baseline uses
[`FacebookAI/roberta-large-mnli`](https://huggingface.co/FacebookAI/roberta-large-mnli).
Pull both via `scripts/download_models.sh`.

## Reproducing the paper

Each numbered table has a one-shot reproduction script:

```bash
bash scripts/reproduce_table5.sh   # routing headline: 0.868 All-acc
bash scripts/reproduce_phase0.sh   # phase-0 follow-ups (XGBoost: +3.24pp)
```

Each script validates the reproduced headline against the paper target
and exits non-zero if outside ±0.01. See [REPRODUCE.md](REPRODUCE.md)
for the full table-by-table breakdown, expected runtimes, and output
locations.

## Documentation

- [INSTALL.md](INSTALL.md) — environment setup, GPU + glibc notes
- [USAGE.md](USAGE.md) — user walkthrough, hint maps for new domains
- [REPRODUCE.md](REPRODUCE.md) — paper reproduction, table by table
- [CITATION.cff](CITATION.cff) — how to cite
- [evaluation/gold/README.md](evaluation/gold/README.md) — gold data
  provenance, schema, licensing
- [evaluation/gold/annotation-protocol.md](evaluation/gold/annotation-protocol.md) —
  the protocol our annotators followed

## Limitations

The released metric is trained and evaluated on English M2T outputs
across the E2E, RNNLG, ViGGO, and Taskmaster domain families; we have
not tested cross-language transfer. Every new domain requires writing
a hint map (a small JSON describing the slot schema), but we have not
measured zero-shot extraction on schemas absent from training. The
1,000-example Eval-2 PERSONAGE gold set is single-annotator from the
paper's author — we report no inter-annotator agreement, and a
secondary annotation pass is planned for v2. The released ranker uses
greedy probability-weighted scoring at inference and was trained with
binary relevance labels; we have not explored more elaborate
preference-learning objectives.

## License

Apache-2.0 for code and gold annotations (see [LICENSE](LICENSE)). The
derived evaluation data under `data/` is CC BY-SA 4.0 — see
[data/README.md](data/README.md) for provenance and attribution. The
LoRA adapters inherit the
[Llama 3 Community License](https://www.llama.com/llama3/license/);
underlying PERSONAGE outputs are publicly distributed. Algorithm
credits for the rule-based aligners (Dušek for E2E, Wen et al. for
RNNLG, Juraska's slug2slug for ViGGO) live in each
`xdomain_ser/baselines/*_aligner.py` module docstring.

## Citation

See [CITATION.cff](CITATION.cff). A DOI will be added once the GEM
proceedings publish.

## Acknowledgments

Parts of this codebase were developed with Claude Code assistance.

## Contact

Author: Davan Harrison (UC Santa Cruz Natural Language & Dialogue
Systems Lab; advisor: Marilyn Walker). Issues and questions:
[`Vrindiesel/xdomain-ser` on GitHub](https://github.com/Vrindiesel/xdomain-ser/issues).
