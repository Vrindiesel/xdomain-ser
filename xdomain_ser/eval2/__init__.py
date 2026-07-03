"""Eval-2 pipeline: human-annotated SER comparison on PERSONAGE outputs.

Stage-2 evaluation paradigm from the GEM 2026 paper: instead of measuring
SER-method agreement against rule-based aligner outputs (Eval 1), this
pipeline measures agreement against human-annotated gold SER labels on
~1000 stratified PERSONAGE-style examples (500 GPT-4o + 500 seq2seq,
balanced per personality).

Module layout (data flow left-to-right):

* :mod:`xdomain_ser.eval2.personage_inference` -- LoRA extraction on
  PERSONAGE-style outputs (produces ``pred_mr`` + ``pred_scores`` for
  ``sample.py`` to consume).
* :mod:`xdomain_ser.eval2.sample` -- stratified 1000-example sampler.
* :mod:`xdomain_ser.eval2.create_files` -- emits TSV + structured-text
  annotation files for human annotators.
* :mod:`xdomain_ser.eval2.process` -- parses completed annotations into
  ``gold-annotated.json``.
* :mod:`xdomain_ser.eval2.create_dataset` -- augments gold with negative
  MRs at labels 0-4 -> ``eval-negatives.json``.
* :mod:`xdomain_ser.eval2.compare` -- three-way SER comparison
  (E2E Aligner / LoRA / NLI) -- **the central script**.
* :mod:`xdomain_ser.eval2.tables` -- generates 5 TSV breakdown tables.
* :mod:`xdomain_ser.eval2.significance` -- McNemar + paired-permutation
  tests across method pairs.

The shipped gold data is at ``evaluation/gold/`` (see
``evaluation/gold/README.md`` for provenance + license).
"""
