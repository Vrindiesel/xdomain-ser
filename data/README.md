# Data

Provenance and licensing for the shipped data. The code in this
repository is Apache-2.0; the derived example data below is not, see
the Licensing section.

## What's here

- `multi_ser_v9/` — the released v9 multi-domain SER splits
  (`train-200.json`, `dev-repr-120.json`, `test.json`), the hint-map
  table (`hint_maps_v4.json`), the few-shot prompt pools, and the
  `files.json` merge manifest. Built from the four public corpora by
  `xdomain_ser.data_prep`; `scripts/build_dataset.sh` reproduces the
  v9 files byte-for-byte (selection `random`, seed 2323).
- `ranking_eval/` — the canonical Eval-2 negatives file and its
  ranker-scored counterpart, consumed by `scripts/reproduce_table5.sh`
  and the corrected-protocol pipeline. Besides corpus-derived text and
  MRs, these embed published model outputs (10-beam extractor
  candidates and ranker scores); see the provenance notes in
  [REPRODUCE.md](../REPRODUCE.md).
- `e2e-nlg/`, `viggo-v1/`, `rnnlg/`, `taskmaster/` — the per-domain
  input hint maps (`hint-map.json`) consumed by
  `scripts/build_dataset.sh`. These schema descriptions are original
  to this project (Apache-2.0). The raw corpora themselves are NOT
  shipped; `build_dataset.sh` documents where to download them.

## Licensing

The example texts and meaning representations in `multi_ser_v9/` and
`ranking_eval/` derive from four public corpora:

- E2E NLG Challenge data — CC BY-SA 4.0
  (https://github.com/tuetschek/e2e-dataset)
- ViGGO — CC BY-SA 4.0 (https://huggingface.co/datasets/GEM/viggo)
- RNNLG benchmark data — released with the RNNLG toolkit, Apache-2.0
  (https://github.com/shawnwun/RNNLG)
- Taskmaster (TM-1/2/3) — CC BY 4.0
  (https://github.com/google-research-datasets/Taskmaster)

Because share-alike applies to the E2E- and ViGGO-derived portions,
the derived data files in `multi_ser_v9/` and `ranking_eval/` are
distributed under **CC BY-SA 4.0**, not the repository's Apache-2.0.
CC BY 4.0 (Taskmaster) and Apache-2.0 (RNNLG) material may be
incorporated into a CC BY-SA 4.0 distribution with attribution, which
this file provides.

## Attribution

- Novikova, Dušek & Rieser (2017). "The E2E Dataset: New Challenges
  for End-to-End Generation." SIGDIAL.
- Juraska, Bowden & Walker (2019). "ViGGO: A Video Game Corpus for
  Data-to-Text Generation in Open-Domain Conversation." INLG.
- Wen, Gašić, Mrkšić, Su, Vandyke & Young (2015). "Semantically
  Conditioned LSTM-based Natural Language Generation for Spoken
  Dialogue Systems." EMNLP.
- Byrne, Krishnamoorthi, Sankar, Neelakantan, Goodrich, Duckworth,
  Yavuz, Dubey, Kim & Cedilnik (2019). "Taskmaster-1: Toward a
  Realistic and Diverse Dialog Dataset." EMNLP.
