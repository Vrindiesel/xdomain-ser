# Reproduce

Each numbered table or figure in the GEM 2026 paper has a one-shot
reproduction script in `scripts/`. The two headline scripts
(`reproduce_table5.sh`, `reproduce_phase0.sh`) validate their headline
number against the paper target and exit non-zero if outside `±0.01`;
the other scripts write the result tables for side-by-side comparison
(`reproduce_table6.sh` additionally prints per-cell |delta| against the
published values). This document is the runbook: what each script
reproduces, runtime budget, dependencies, and where to find the output.

## Conventions

- All scripts must be run from the repo root with the conda env
  activated (see [INSTALL.md](INSTALL.md)).
- Models are pulled into `models/extractor/` and `models/ranker/` by
  `scripts/download_models.sh` (called by the scripts that run LoRA
  inference: `reproduce_table3.sh`, `reproduce_table4.sh`, and
  `reproduce_table5.sh --rescore`; the remaining scripts consume the
  shipped predictions and load no LoRA weights).
- Results land in `evaluation/results/<table>/` as TSV + JSON.
- Reference runtimes are wall-clock on an NVIDIA RTX 3090 (24 GB).
- All routing experiments use `seed=42`; numbers are deterministic.

## Quick reference

| Script | Reproduces | Headline | Runtime | Extras |
|---|---|---|---|---|
| `reproduce_table3.sh` | PBL (GPT-4o) vs LoRA extraction | per-domain SER + slot-F1 tables | ~2 h | `requirements-pbl.txt` + `OPENAI_API_KEY` |
| `reproduce_table4.sh` | OGR effect: greedy vs beam-10 + ranker | per-domain SER tables | ~18 h (measured; see note) | — |
| `reproduce_table5.sh` | SER agreement with per-example routing | LR-Routing All-acc = **0.868** | ~5 min | — |
| `reproduce_table6.sh` | vs rule-based SER tools on the six Eval-2 topics | Table 6 rule + learned rows, with paper deltas | ~2 min (CPU; run table5 first) | — |
| `reproduce_table7.sh` | PERSONAGE robustness (five-way, per-source) | Table 7 LLM/Seq2Seq columns (exact except LR ±1–7 pp; see section) | ~1 min | — |
| `reproduce_appendix_gold_comparison.sh` | gold three-way comparison (aligner / LoRA / NLI) | five breakdown tables | ~5 min | — |
| `reproduce_appendix_d.sh` | 5-way routing on PERSONAGE gold | five-way comparison + oracle ceiling | ~5 min | — |
| `reproduce_phase0.sh` | Phase-0 follow-ups (XGBoost routing) | XGB-CW All-acc = **0.9004** (+3.24pp over LR) | ~5 min (XGBoost), +10 min (DeBERTa), +10 min (value-norm) | `requirements-experimental.txt` |

The "Headline" column is each script's key output; `reproduce_table5.sh`
and `reproduce_phase0.sh` gate on theirs with `±0.01` and exit non-zero
outside it. The "Runtime" column is wall-clock on RTX 3090; CPU is too
slow for any of these except `reproduce_table6.sh` (CPU-only) and
`reproduce_table5.sh`'s post-NLI evaluation stage.

## What each script recomputes (and what it consumes)

Reproduction here means recomputation: every number a script prints is
computed at runtime. The shipped `evaluation/results/` directories hold
the published reference outputs; each script recomputes its numbers and
writes them to the same locations. Two released data files do carry
cached model predictions from the published runs, and the table below
makes explicit where those are consumed instead of re-running the
underlying model:

| Script | Runs live | Consumes shipped predictions |
|---|---|---|
| `reproduce_table3.sh` | GPT-4o PBL + LoRA extraction, SER eval | — |
| `reproduce_table4.sh` | greedy + beam-10 extraction, ranker scoring, SER eval | — |
| `reproduce_table5.sh` | NLI inference, dev/test split, threshold sweep, LR fit, all metrics (`--rescore` adds live ranker scoring) | beam candidates + ranker scores in the eval file (default mode) |
| `reproduce_table6.sh` | all three rule-based aligners, per pair; table assembly | learned rows from *your* `reproduce_table5.sh` run; beams as above |
| `reproduce_table7.sh`, `reproduce_appendix_d.sh` | NLI inference, E2E aligner, threshold sweep, LR fit, all metrics | beam candidates + ranker scores embedded in `gold-annotated.json` |
| `reproduce_appendix_gold_comparison.sh` | NLI inference, E2E aligner, all metrics | same |
| `reproduce_phase0.sh` | NLI inference, feature build, XGBoost sweep + fit | beam candidates + ranker scores in the eval file |

Two provenance notes:

- **Beam candidates are part of the dataset, not a cache.** The 9,042
  Eval-2 pairs were constructed around the published 10-beam extractor
  candidates (`pred_mr`); regenerating beams would change the evaluation
  set itself, so no script ever does.
- **Ranker scores are recomputable — and verified.** `reproduce_table5.sh
  --rescore` re-runs the LoRA ranker over the cached beams with the
  published per-text protocol (~75 min on RTX 3090) instead of consuming
  the shipped `pred_scores`; on the reference RTX 3090 it regenerates the
  shipped scores to within 2e-5 with zero top-1 selection changes and
  reproduces the headline exactly. Compute dtype matters here: candidate
  scores sit on near-ties (53% of texts have a top-2 score gap < 0.02),
  so the scorer's faster bf16 default flips ~32% of top-1 selections and
  moves selection-based metrics by several pp — `--rescore` therefore
  pins `--compute_dtype float32`, the dtype the published runs used.

## Hardware budget

The full set takes roughly a GPU-day on a single RTX 3090 if you run
everything; `reproduce_table4.sh` dominates at ~18 h measured (beam-10
extraction 9.5 h + fp32 ranker scoring 7 h), with `reproduce_table3.sh`
adding ~2 h. Treat Table 4 as an overnight-plus-a-morning run. The scorer's faster bf16 mode is NOT
used for reproduction: candidate scores sit on near-ties and bf16 flips
~32% of top-1 selections, which moves the ranker-selected rows away from
the published values (see "What each script recomputes"). For a sanity
check that the pipeline
works end-to-end, run `reproduce_table5.sh` first — it takes about
5 minutes and exercises the routing pipeline that the paper's
headline number comes from.

Peak GPU memory:

- ~7 GB during LoRA inference (4-bit Llama-3.2-3B + adapter)
- ~1.5 GB during NLI inference (RoBERTa-large-MNLI)
- ~9 GB during routing scripts when both are loaded sequentially

A 16 GB card handles all scripts at default batch sizes; an 8 GB card
needs `--batch_size 4` for the LoRA-inference paths.

## Table 5 — routing headline (smoke-test target)

```bash
bash scripts/reproduce_table5.sh
```

This runs `xdomain_ser.routing.selector` on the canonical scored eval
file `data/ranking_eval/negatives-v6-test.200.pe5.b10.rexp4.ckpt60.scores.json`,
performs a topic-stratified 50/50 dev/test split with `seed=42`,
trains the logistic-regression router on the dev split, and evaluates
all four methods (LoRA / NLI / score-threshold / LR-routing) on the
held-out test split.

By default the ranker scores shipped inside the eval file are used. To
re-run the LoRA ranker itself over the cached beam candidates first
(published per-text protocol at fp32 compute, ~75 min extra):

```bash
bash scripts/reproduce_table5.sh --rescore
```

Expected output:

```
Reproduced LR-Routing All_acc: 0.8680709534368071
Paper target:                  0.868 (+/- 0.01)
|reproduced - target|:         0.0001
PASS: within tolerance.
```

Other methods in the same run:

| Method | All_acc | SER_MAE |
|---|---|---|
| LoRA | 0.7641 | 0.0776 |
| NLI | 0.8051 | 0.0495 |
| ScoreRouting | 0.8614 | 0.0350 |
| **LR-Routing** | **0.8681** | 0.0363 |
| (Oracle) | 0.9073 | 0.0265 |

Outputs in `evaluation/results/table5/`:

- `comparison_results.tsv` — overall + per-source breakdown.
- `per_topic_comparison.tsv` — per-topic breakdown for all four methods.
- `score_routing_sweep.tsv` — threshold sweep for the score-routing baseline.
- `lr_feature_importance.tsv` — LR coefficients (top features: `top_score`,
  `min_nli_prob`, `nli_coverage`).
- `routing_details.json` — per-example routing decisions for downstream
  analysis.

## Phase-0 — XGBoost routing follow-up

```bash
bash scripts/reproduce_phase0.sh                       # XGBoost only (default)
PHASE0_DEBERTA=1 bash scripts/reproduce_phase0.sh      # + DeBERTa NLI swap
PHASE0_VALUENORM=1 bash scripts/reproduce_phase0.sh    # + value normalisation
```

The XGBoost routing reproducer always runs and is the script's
validation target. The DeBERTa and value-normalisation variants are
opt-in via env vars; both take ~10 min each and require additional
deps (`sentence-transformers` for value normalisation).

Expected headline:

```
Reproduced XGBoost-CW All_acc: 0.9004
Paper target:                  0.9004 (+/- 0.01)
PASS
```

Top feature importances by gain (from `feature_importance.tsv`):

| Feature | Importance (gain) |
|---|---|
| `top_score` | 0.342 |
| `min_nli_prob` | 0.192 |
| `nli_coverage` | 0.092 |
| `topic_hotel` | 0.079 |
| `slot_ratio` | 0.068 |
| `n_gold_slots` | 0.062 |

Outputs in `evaluation/results/phase0/`:

- `xgboost/results_comparison.tsv` — XGBoost variants + LR baseline.
- `xgboost/best_config.json` — winning hyperparameter config.
- `xgboost/feature_importance.tsv` — feature gain ranking.
- (with `PHASE0_DEBERTA=1`) `deberta/nli_comparison.tsv` — RoBERTa vs
  DeBERTa-v3-large NLI head-to-head. **Note:** DeBERTa underperforms
  RoBERTa on our slot templates (documented negative result).
- (with `PHASE0_VALUENORM=1`) `value_norm/value_norm_analysis.tsv` —
  per-slot SBERT + NLI cross-verification scores.

## Table 6 — vs rule-based SER tools (six Eval-2 topics)

```bash
bash scripts/reproduce_table5.sh   # prerequisite: learned per-topic rows
bash scripts/reproduce_table6.sh
```

Compares the learned methods against the three published rule-based SER
tools (E2E script, RNNLG, ViGGO) on the six Eval-2 topics where rule
tools exist. The rule rows are computed by driving the verbatim aligner
code over the scored Eval-2 file; the learned rows come from Table 5's
`per_topic_comparison.tsv`. CPU-only, ~2 minutes.

Protocol provenance: see the three protocol notes at the top of
`scripts/make_table6.py` (aligner conditioning, full-set vs test-half
basis, and the RNNLG default-domain quirk — all replicated as
published).

Outputs in `evaluation/results/table6/`:

- `table6.tsv` — all rows (rule full-set + test-half, learned methods),
  with `paper_row` marking the published cells.
- `table6.md` — paper rows with |delta| vs the published values, plus
  the protocol notes.

## Table 7 — PERSONAGE robustness

```bash
bash scripts/reproduce_table7.sh
```

Five-way routing comparison (`xdomain_ser.routing.personality`) on the
personality-stratified test split of the gold-annotated set. Table 7's
"LLM Pers." and "Seq2Seq" columns are this run's per-source breakdown
(250 examples each); the "E2E (no style)" column comes from Table 6's
E2E rows.

Reproduction provenance: the published run used precomputed NLI results
from the gold three-way comparison (threshold 0.5), so the script pins
`--nli_threshold 0.5`. Aligner, LoRA, NLI, and ScoreRouting cells
reproduce the paper exactly; LR-Routing refits on NLI probability
features computed with today's 138-template set (the paper's runs used
133) and lands within ~1–7 pp of the published cells.

Outputs in `evaluation/results/table7/`:

- `per_source_comparison.tsv` — Table 7's LLM/Seq2Seq columns.
- `per_personality_comparison.tsv`, `comparison_results.tsv`,
  `score_routing_sweep.tsv`, `lr_feature_importance.tsv`,
  `routing_details.json`.

## Appendix — gold three-way comparison

```bash
bash scripts/reproduce_appendix_gold_comparison.sh
```

Same pipeline as Table 7 into its own results directory
(`evaluation/results/appendix_gold_comparison/`), kept as an
independent entry point for the gold-set comparison. (This script was
formerly mis-titled `reproduce_table6.sh`.)

## Appendix D — 5-way routing on PERSONAGE gold

```bash
bash scripts/reproduce_appendix_d.sh
```

Runs `xdomain_ser.routing.personality` — the five-way comparison
between the rule-based aligner, LoRA MySER, NLI, score-threshold
routing, and LR routing, on the personality-stratified dev/test split
of the gold-annotated data. The NLI threshold is pinned to 0.5 to match
the published precomputed-NLI path (see the Table 7 section above for
the reproduction provenance and the known ~1–7 pp LR-Routing residual
from NLI template-set drift).

Outputs in `evaluation/results/appendix_d/`:

- `comparison_results.tsv` — five-way overall comparison.
- `per_source_comparison.tsv` — LLM vs seq2seq breakdown.
- `per_personality_comparison.tsv` — by personality type.
- `score_routing_sweep.tsv` — score-threshold sweep on dev.
- `lr_feature_importance.tsv` — LR coefficients (personality variant
  has pers / source one-hots instead of topic).
- `routing_details.json` — per-example routing decisions.

## Tables 3 and 4 — extraction quality (long runs)

These two scripts re-run inference over the full multi-domain test set
and take several hours each.

### Table 3 — PBL vs LoRA

```bash
export OPENAI_API_KEY=sk-...
pip install -r requirements-pbl.txt
bash scripts/reproduce_table3.sh
```

Runs the GPT-4o PBL extractor and the LoRA extractor on the same
multi-domain test set, then scores both with
`xdomain_ser.extraction.eval`. PBL inference uses ~50 K test calls and
takes about an hour at the OpenAI rate limit. LoRA inference takes
another hour for the full set.

Outputs in `evaluation/results/table3/`:

- `pbl_predictions.json`, `lora_predictions.json` — raw outputs.
- `pbl_scores.json`, `lora_scores.json` — per-topic SER + slot-F1.

### Table 4 — OGR effect

```bash
bash scripts/reproduce_table4.sh
```

Three LoRA inference passes (measured wall-clocks, RTX 3090,
2026-06-12 verification run):

1. Greedy single-beam over the test set (1 h 28 m).
2. Beam-10 over the test set (9 h 21 m).
3. Ranker scoring over the beam-10 outputs (fp32, 7 h 00 m).

Then scores each pass three ways: as-is, top-of-beam, ranker-selected
(~10 min total). The Table 4 numbers compare the three. ~18 h end to end.

Protocol notes from the verification run:

- **Effective eval set: 12,144 examples — the paper's 12 test topics.**
  `test.json` carries two extra hint_map_ids (`hm_tm2_hotels`, 4,184
  examples, and 7 `hm_tm1_movie_ticket` stragglers) with no entry in the
  shipped hint-map table; inference skips them, reproducing the
  published 12-topic protocol.
- **Table 4's "SER-Acc" column is 1 − micro-SER** (from
  `ser_score.SER` in the eval outputs); "F1" is `slot_f1.f1`.
- Expected LoRA-row values from this pipeline vs published:

  | | F1 | SER-Acc | F1 (R) | SER-Acc (R) |
  |---|---|---|---|---|
  | published | .819 | .75 | .918 | .88 |
  | this pipeline | .854 | .795 | .927 | .900 |

  The ranked row reproduces within ~1–2 pp; the greedy row lands
  ~3.5–4.5 pp above published. Likely cause: the shipped few-shot pool
  is `prompt-examples-dev-repr.json` (the representative subset built
  by `build_dataset.sh`) while the paper's runs drew prompts from the
  full `prompt-examples-dev.json` pool. The OGR effect itself
  reproduces (ranking adds +7.3 pp F1 / +10.5 pp SER-Acc here vs
  +9.9 / +13 published).

Outputs in `evaluation/results/table4/`:

- `greedy_predictions.json`, `beam10_predictions.json`,
  `beam10_scored.json` — the three inference outputs.
- `greedy_scores.json`, `beam10_topbeam_scores.json`,
  `beam10_ranked_scores.json` — the three SER evaluations.

## Output directory layout

After running all reproductions, `evaluation/results/` looks like:

```
evaluation/results/
├── table3/
│   ├── {pbl,lora}_predictions.json
│   └── {pbl,lora}_scores.json
├── table4/
│   ├── {greedy,beam10,beam10_scored}_predictions.json
│   └── {greedy,beam10_topbeam,beam10_ranked}_scores.json
├── table5/
│   ├── comparison_results.tsv         ← headline numbers
│   ├── per_topic_comparison.tsv
│   ├── score_routing_sweep.tsv
│   ├── lr_feature_importance.tsv
│   └── routing_details.json
├── table6/
│   ├── table6.tsv
│   └── table6.md
├── table7/
│   └── (same five-way outputs as appendix_d/; see Table 7 section)
├── appendix_gold_comparison/
│   ├── comparison.json
│   └── tables/method-comparison-*.tsv
├── appendix_d/
│   └── (see Appendix D section above)
└── phase0/
    ├── xgboost/
    │   ├── results_comparison.tsv     ← XGBoost headline
    │   ├── best_config.json
    │   └── feature_importance.tsv
    ├── deberta/   (only if PHASE0_DEBERTA=1)
    └── value_norm/  (only if PHASE0_VALUENORM=1)
```

## When a reproduction misses tolerance

The scripts exit non-zero if the reproduced headline lands outside
`±0.01` of the paper target. Common causes:

- **Wrong checkpoint.** If you didn't run `scripts/download_models.sh`
  and instead symlinked an arbitrary local checkpoint, the numbers will
  differ. The paper-trained adapters live on the HF Hub at
  `DavanHarrison/xdomain-ser-extractor` and `DavanHarrison/xdomain-ser-ranker`.
- **Different seed.** All routing scripts use `seed=42`. Override with
  `--seed N` to explore variance.
- **Numerical / package-version drift.** Transformers, peft, and
  bitsandbytes versions matter; the pins in `requirements.txt` are the
  versions used to generate the paper numbers.
- **GPU non-determinism.** Inference with greedy decoding is
  deterministic; beam search and sampling are not. We do not enable
  CUDA deterministic mode by default — if you need bit-exact runs, set
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` and call
  `torch.use_deterministic_algorithms(True)` at program start.

If a reproduction lands well outside tolerance (say, > 5%), open an
issue with the script output and your GPU + driver + transformers
version.
