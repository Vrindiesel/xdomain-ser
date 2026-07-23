# Usage

This document walks through using the `xdomain-ser` CLI and Python API
for the three most common tasks:

1. Scoring an existing M2T model's outputs against a known gold MR
2. Extracting an MR from text (without comparison)
3. Writing a hint map so the metric can score outputs in a new domain

For installation, see [INSTALL.md](INSTALL.md). For reproducing the
paper's table-by-table results, see [REPRODUCE.md](REPRODUCE.md).

## CLI overview

The CLI — run as `python -m xdomain_ser.cli` from the repo root —
exposes six subcommands:

| Subcommand | Purpose |
|---|---|
| `score` | Extract MR from text and compute SER + slot-F1 vs gold MR. |
| `extract` | Extract MR from text only (no gold required). |
| `rank` | Score `k` candidate MRs against text+gold with the trained ranker. |
| `nli` | NLI-based slot verification: which gold slots does the text entail? |
| `route` | LoRA extraction + NLI verification + score-threshold routing on a single example. |
| `reproduce` | Wrap `scripts/reproduce_table_N.sh`. |

Every subcommand accepts `--text` / `--gold-mr` / `--domain` for
single-example mode, plus `--input-file <json>` / `--limit N` for
batch mode.

Run `python -m xdomain_ser.cli <subcommand> --help` for the full flag set.

## Single-example walkthrough

```bash
python -m xdomain_ser.cli score \
  --text "Cotto is a coffee shop near The Portland Arms that serves English food at a high price with average customer rating." \
  --gold-mr '{
      "name": "Cotto",
      "eatType": "coffee shop",
      "food": "English",
      "priceRange": "high",
      "customerRating": "average",
      "near": "The Portland Arms"
    }' \
  --domain e2e
```

The CLI loads the LoRA extractor on first invocation (~12 s),
generates the predicted MR with greedy decoding, translates it back to
E2E-native slot names, and computes both metrics against the gold.

Output:

```
=== xdomain-ser score ===
Text:           Cotto is a coffee shop near The Portland Arms that serves English food at a high price with average customer rating.
Domain:         e2e  (hm_e2e_nlg)
Gold MR:        {"name": "Cotto", "eatType": "coffee shop", ...}
Predicted MR:   {"name": "Cotto", "eatType": "coffee shop", ...}

  SER:          0.0000  (S=0  D=0  I=0  N_ref=6)
  Slot F1:      1.0000  (precision=1.0000, recall=1.0000)
  TP/FP/FN:     6/0/0
```

The predicted MR matches the gold exactly here — `SER = 0` and slot
F1 = 1.0. For a less complete gold MR, see the README quickstart.

### Interpreting the output

- **SER** = (S + D + I) / N_ref, where N_ref is the number of slots in
  the gold MR. Lower is better, ≥ 1 means there are at least as many
  errors as gold slots.
- **S** (substitutions), **D** (deletions), **I** (insertions) are the
  three error categories. S counts slots present in both but with
  different values; D counts slots only in gold; I counts slots only in
  the prediction.
- **Slot F1** treats slot extraction as a classification problem: a
  true positive is a slot in both with matching values. Lists are
  compared as sets, with optional containment matching.

## Batch mode

For larger evaluations, pass a JSON array of examples:

```bash
python -m xdomain_ser.cli score \
  --input-file my_outputs.json \
  --output-file /tmp/scores.json \
  --domain e2e \
  --limit 50
```

The input JSON is a list of objects with at minimum `text` and `mr`:

```json
[
  {
    "text": "...",
    "mr": {"name": "Cotto", "priceRange": "high"},
    "hint_map_id": "hm_e2e_nlg"
  },
  ...
]
```

`hint_map_id` per-example overrides the `--domain` flag (useful when
batching across domains). `--limit N` truncates to the first N records.
`--output-file` writes per-example results plus aggregate
mean-SER / mean-slot-F1 in a JSON we can re-load later.

## Supported domains out of the box

The CLI's `--domain` flag accepts these short aliases:

| Alias | hint_map_id | Source |
|---|---|---|
| `e2e` | `hm_e2e_nlg` | E2E NLG Challenge |
| `viggo` | `hm_viggo` | ViGGO video games |
| `rnnlg_hotel` | `hm_rnnlg_hotel` | RNNLG hotel |
| `rnnlg_laptop` | `hm_rnnlg_laptop` | RNNLG laptop |
| `rnnlg_restaurant` | `hm_rnnlg_restaurant` | RNNLG restaurant |
| `rnnlg_tv` | `hm_rnnlg_tv` | RNNLG TV |
| `tm1_auto_repair` | `hm_tm1_auto_repair_appt` | Taskmaster-1 auto repair |
| `tm1_coffee_ordering` | `hm_tm1_coffee_ordering` | Taskmaster-1 coffee |
| `tm1_movie_tickets` | `hm_tm1_movie_tickets` | Taskmaster-1 movie tickets |
| `tm1_pizza_ordering` | `hm_tm1_pizza_ordering` | Taskmaster-1 pizza |
| `tm1_restaurant_table` | `hm_tm1_restaurant_table` | Taskmaster-1 restaurant |
| `tm1_uber_lyft` | `hm_tm1_uber_lyft` | Taskmaster-1 Uber/Lyft |

Pass `--domain hm_<id>` directly for any other domain present in
`data/multi_ser_v9/hint_maps_v4.json`.

## Writing a hint map for a new domain

The metric's "knowledge" of a domain lives in a hint map — a small JSON
file describing the slot schema. To score outputs from a domain not in
the table above, write a hint map and point `--hint-map-path` at it.

### Schema

A hint map JSON looks like:

```json
{
  "hm_my_domain": {
    "hint_map_id": "hm_my_domain",
    "hint_map": {
      "slot_name_1": "human-readable description of the slot and its likely values",
      "slot_name_2": "...",
      "...": "..."
    }
  }
}
```

The top-level keys are hint_map_ids (we prefix them with `hm_` by
convention). Each hint_map_id maps to an object with:

- `hint_map_id` (str): the same id, repeated so downstream code can
  recover it from the entry alone.
- `hint_map` (dict): slot name → free-text description. The
  description goes into the LoRA prompt as part of the domain schema
  hint, so write it as if you were briefing a human annotator.

Optional fields:

- `value_map` (dict): slot name → list of valid enumerated values.
  Used for domains where the LoRA's training data had a closed value
  set (e.g., ViGGO ratings). Omit if values are open-ended.

### Worked example: a smart-thermostat schedule domain

Suppose we want to score M2T outputs that describe smart-thermostat
schedules. Each MR has a target temperature, an HVAC mode, and a
time-of-day. We can write the hint map directly:

```json
{
  "hm_thermostat": {
    "hint_map_id": "hm_thermostat",
    "hint_map": {
      "target_temp": "target temperature in Fahrenheit (e.g., 68, 72)",
      "hvac_mode": "HVAC mode (heat, cool, auto, off)",
      "time_of_day": "schedule slot (morning, afternoon, evening, night)",
      "zone": "house zone (living room, bedroom, basement, whole house)"
    }
  }
}
```

Save as `my_hint_maps.json`. Then:

```bash
python -m xdomain_ser.cli score \
  --text "Set the bedroom to 68 in heat mode at night." \
  --gold-mr '{"target_temp":"68","hvac_mode":"heat","time_of_day":"night","zone":"bedroom"}' \
  --domain hm_thermostat \
  --hint-map-path my_hint_maps.json
```

The LoRA wasn't fine-tuned on thermostat data, so extraction quality
on a fresh domain depends on how well the schema matches the patterns
the model learned across the seven training-time domains. In practice
slot names that look like the training schemas (free-text values like
names and locations) transfer better than ones with enumerated values
the model has never seen.

### Hint maps that match the LoRA's training schemas

If you control the schema, looking like one of the training domains
helps a lot. The seven domain families the LoRA learned from cover:

- Restaurant / venue attributes (name, eatType, food, area, near,
  priceRange, customerRating, familyFriendly)
- Hotel attributes (name, area, price range, internet, parking, pets)
- Consumer products with specs (laptop / TV: name, family, price, weight,
  resolution, accessories)
- Video game metadata (developer, genre, platforms, ESRB rating,
  multiplayer support)
- Service-request task data (auto repair, coffee ordering, movie
  tickets, pizza, restaurant reservations, Uber/Lyft)

A hint map for cocktails, books, or vacation rentals likely transfers
well because the slot shapes (name, type, descriptive attributes) match.
A hint map for protein-folding parameters or air-traffic clearances
probably does not.

## Python API

For programmatic use, instantiate the CLI helpers directly:

```python
from pathlib import Path
from xdomain_ser.core import data_helper as dh, ser
from xdomain_ser.cli import (
    _load_extractor,
    _extract_one,
    _translate_lora_mr,
    _ser_report,
    _resolve_checkpoint_dir,
    _resolve_hint_map_path,
    _resolve_prompt_examples_path,
)

checkpoint = _resolve_checkpoint_dir(None)
hint_map_path = _resolve_hint_map_path(None)
prompt_examples_path = _resolve_prompt_examples_path(None)

model, tokenizer, hint_map_table, pool = _load_extractor(
    checkpoint, hint_map_path, prompt_examples_path
)

text = "Cotto is a coffee shop ..."
pred_str = _extract_one(text, "hm_e2e_nlg", model, tokenizer,
                        hint_map_table, pool)
pred_dict = dict(ser.extract_attributes_dict(pred_str.replace(dh.LIST_END, "")))
pred_native = _translate_lora_mr(pred_dict, "e2e")

gold = {"name": "Cotto", "eatType": "coffee shop", ...}
report = _ser_report(pred_native, gold)
print(report)  # {"SER": ..., "S": ..., "D": ..., "I": ..., "slot_f1": ...}
```

The `_STATE` cache inside `xdomain_ser.cli` keys on the
(checkpoint, hint_map, prompt_examples) tuple, so subsequent calls in
the same process don't re-load the model.

For raw SER computation without going through the LoRA, use
`xdomain_ser.core.ser` directly:

```python
from xdomain_ser.core import ser

result = ser.compute_ser(
    pred={"name": "Cotto", "priceRange": "high"},
    ref={"name": "Cotto", "priceRange": "high", "near": "river"},
)
# {"SER": 0.3333, "S": 0, "D": 1, "I": 0, "N_ref": 3}
```

## Other subcommands

### `extract` — extraction only

```bash
python -m xdomain_ser.cli extract \
  --text "Cotto is a coffee shop ..." \
  --domain e2e
```

Returns the LoRA's predicted MR without a gold comparison. Useful when
you don't have a gold MR but want to inspect what the metric "sees".

### `nli` — NLI slot verification

```bash
python -m xdomain_ser.cli nli \
  --text "Cotto is a coffee shop near The Portland Arms ..." \
  --gold-mr '{"name":"Cotto","near":"The Portland Arms"}' \
  --threshold 0.3
```

Loads RoBERTa-MNLI, builds one hypothesis per gold slot, and prints the
per-slot entailment probability plus the recovered MR (slots with prob
> `--threshold`). Good for diagnosing whether a model output "really
says" each gold slot.

### `route` — LoRA + NLI + threshold routing

```bash
python -m xdomain_ser.cli route \
  --text "Cotto is a coffee shop ..." \
  --gold-mr '{...}' \
  --domain e2e \
  --score-threshold 2.85
```

Runs the LoRA extraction (with beam-10 + ranker), the NLI verification,
and decides which to trust based on the ranker's top score. Prints both
methods' MRs plus the routed choice. Useful for understanding when the
metric is confident vs uncertain.

### `rank` — score k candidates

```bash
python -m xdomain_ser.cli rank \
  --input-file candidates.json \
  --output-file ranked.json
```

Takes a JSON of `{text, mr, hint_map_id, pred_mr: [...]}` rows (the
output of `python -m xdomain_ser.cli extract --num-beams 10`) and adds a
`pred_scores` field with the ranker's probability-weighted score for
each candidate.

## When to use which method

- **LoRA alone (`extract` or `score` with single-beam)** — fastest;
  good when the LoRA was trained on a similar domain.
- **LoRA + ranker (`score` with `--num-beams 10`)** — slightly slower,
  noticeably more robust when the single-beam output is borderline.
- **NLI alone (`nli`)** — slower (RoBERTa-MNLI per slot) but
  domain-agnostic; useful when the LoRA misfires on out-of-distribution
  text.
- **Routing (`route`)** — the headline 0.868 setup. Use when you have
  both methods available and want the strongest result.

## Common pitfalls

**The LoRA returns slots in "internal" names.** The extractor speaks a
multi-domain ontology (`venue_type`, `cuisine_type`, `nearby_landmark`,
…) rather than the E2E-native names (`eatType`, `food`, `near`, …). The
CLI translates back to native names for the E2E domain automatically.
For other domains the LoRA-internal names match the native names, so no
translation is needed. If you call the Python API directly, use
`xdomain_ser.cli._translate_lora_mr(pred, domain)` to apply the same
mapping.

**`familyFriendly` value translation.** In E2E, `familyFriendly` takes
`yes`/`no`; the LoRA emits `family-friendly` / `not-family-friendly`.
The CLI translates this for the E2E domain.

**`<LIST>...</LIST>` parse quirk.** The MR parser strips `<LIST>(` at
the start but treats `</LIST>` (all caps) inconsistently with the
mixed-case `</List>`. The CLI replaces `</LIST>` before parsing; if
you write your own pipeline, strip the closing tag explicitly.

**Empty gold MRs.** SER divides by `N_ref` and clamps the denominator
to 1, so an empty gold MR with no predicted slots gives `SER = 0`. An
empty gold with predicted slots gives `SER = I` (all insertions).
