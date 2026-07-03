"""Rule-based slot aligners used as Stage-7 baselines.

These three modules implement domain-specific SER evaluators based on
keyword-matching alignment:

* :mod:`xdomain_ser.baselines.e2e_aligner` -- E2E NLG Challenge
* :mod:`xdomain_ser.baselines.rnnlg_aligner` -- RNNLG hotel / laptop /
  restaurant / TV
* :mod:`xdomain_ser.baselines.viggo_aligner` -- ViGGO video games

All three are Davan-authored re-implementations of published alignment
algorithms (Dušek for E2E, Wen et al. for RNNLG, Juraska/slug2slug for
ViGGO -- see each module's docstring for full citation). The Python code
is Apache-2.0; the algorithm credits live in the module docstrings.

Each module exposes the same surface: ``pack_*_nlg_mr(...)`` for MR
canonicalisation, ``extract_mr(text, mr)`` for slot-realisation
extraction, and ``eval_compute_ser(examples, ...)`` for end-to-end SER
on a list of examples.
"""
