# Changelog

## v1.0.0 — unreleased

First public release, accompanying the GEM @ ACL 2026 paper
"Cross-Domain Semantic Fidelity Evaluation for Meaning-to-Text
Generation".

**Eval-2 conditioning correction.** The published Eval-2 routing
evaluation conditioned every method on information unavailable in
deployment: the ranker scored each candidate MR against the example's
true MR in its gold-reference prompt field (the P-oracle protocol),
and NLI verified the true MR's slot-values rather than the
candidate's. The corrected per-pair P-deploy protocol
(`xdomain_ser.routing.pair_selector --protocol deploy`) re-scores
every (example, candidate) pair with only that pair's candidate MR
visible to both methods (`ranking.score --ref_source hypothesis`; NLI
hypotheses built from the candidate's slot-values). Absolute All-acc
drops for every method (LoRA 0.4796, NLI 0.2257, LR-Routing 0.4929,
XGB-Routing 0.5220, oracle ceiling 0.5446 on the same 4,510 test
pairs), and the method ordering changes: NLI and score-threshold
routing degrade most, falling below LoRA alone, while learned routing
(LR, XGBoost) stays ahead of every single method. The published
P-oracle numbers remain reproducible bit-exact via
`--protocol oracle` (LoRA 0.7641, NLI 0.8051, LR-Routing 0.8681).
Full artifacts and run manifests: `evaluation/results/eval2_corrected/`.

- `xdomain_ser` package: cross-domain SER metric (`core.ser`), LoRA
  extraction (`extraction`), over-generate-and-rank candidate scoring
  (`ranking`), RoBERTa-MNLI verification baseline (`nli`), per-example
  LoRA/NLI routing (`routing`: score-threshold, logistic regression, and
  experimental cost-weighted XGBoost), and ports / re-implementations of
  the E2E / RNNLG / ViGGO rule-based SER tools (`baselines`; upstream
  credit and licenses in each module docstring).
- Typer CLI (`python -m xdomain_ser.cli`):
  `score|extract|rank|nli|route|reproduce`.
- LoRA adapters on Hugging Face Hub:
  `DavanHarrison/xdomain-ser-extractor`,
  `DavanHarrison/xdomain-ser-ranker`.
- Reproduction scripts for paper Tables 3–7, Appendix D, and the
  phase-0 XGBoost follow-up, with runtime notes in `REPRODUCE.md`.
- Multi-domain SER dataset (v9) build pipeline (`data_prep`) and the
  1,000-example human-annotated PERSONAGE gold set with its annotation
  protocol (`evaluation/gold/`).
- Unit test suite (pytest), ruff-clean codebase, GitHub Actions CI.
