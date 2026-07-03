# Eval-2 gold-annotated SER data

The 1000-example human-annotated set used in the Eval-2 (PERSONAGE) SER
comparison reported in the GEM @ ACL 2026 paper.

## Files

| File | Size | Purpose |
|---|---|---|
| `gold-annotated.json` | ~3 MB | Primary artifact: 1000 examples with human-annotated SER labels. |
| `eval-negatives.json` | ~5 MB | `gold-annotated.json` augmented with negative MRs at SER label bins 0–4 (one per label per example, where available). |
| `raw/sampled-examples.json` | ~2.7 MB | Stratified 1000-example sample drawn from Chapter-4 outputs *before* annotation. |
| `raw/annotation-data.tsv` | ~370 KB | The TSV the annotators filled in (`has_error` + `error_details` columns). |

## Provenance

The 1000 examples are stratified to 500 LLM (GPT-4o) and 500 seq2seq
(PERSONAGE) outputs, balanced at 100 per personality type per source.
LLM source: Chapter-4 experiments `exp3`–`exp9` (chat-completion-based
personality-conditioned M2T). Seq2seq source: three SV-NLG models
(`features_guide_model1`, `features_token_model7`,
`token_supervision_model27`).

Sampling, ID assignment, and stratification are reproducible from the
upstream outputs via `xdomain_ser.eval2.sample` (the script is shipped
in `xdomain_ser/eval2/sample.py`).

## Schema

Each example in `gold-annotated.json` has:

```jsonc
{
  "id": "llm_exp4_001",                  // unique stable ID
  "source": "llm",                       // "llm" | "seq2seq"
  "experiment": "exp4",                  // upstream experiment / model
  "personality": "AGREEABLE",            // 1 of 5 personality types
  "mr": { ... },                         // reference MR (E2E-native slot names)
  "pred": "...",                         // the M2T model's generated text
  "clean_pred_text": "...",              // post-processed text (annotator input)
  "ref": "...",                          // reference utterance
  "pred_mr": [ ... ],                    // LoRA candidate MR strings (k=10)
  "pred_scores": [ ... ],                // ranker scores for each candidate
  "orig_idx": 42,                        // index in upstream source file
  "has_error": true,                     // human annotator's YES/NO
  "annotation_errors": [                 // structured error list
    {"type": "deletion", "slot": "area"}
  ],
  "gold_ser": {                          // computed from annotation_errors
    "SER": 0.25, "S": 0, "D": 1, "I": 0, "N_ref": 4
  },
  "gold_pred_mr": { ... }                // reconstructed: gold MR with errors applied
}
```

`eval-negatives.json` has the same per-example schema plus a `negatives`
field — a list of `{label, mr, ser_vals}` where `label ∈ {0,1,2,3,4}` is
the SER difficulty bin.

## Licensing

The underlying PERSONAGE outputs (Mairesse & Walker; Oraby et al.) are
already publicly distributed; no additional sign-off is required to
release these annotations alongside them. The annotation labels
(`has_error`, `annotation_errors`, `gold_ser`, `gold_pred_mr`) are
Davan Harrison's work, released under Apache-2.0 with the rest of the
codebase.

See `annotation-protocol.md` for the protocol annotators followed.
