# SER Annotation Protocol

The protocol annotators followed when producing the gold labels in
`gold-annotated.json`. Reproduced verbatim from the annotation file
header (`raw/annotation-data.tsv` and the structured-text version).

## Task

For each example, compare the **MR** (meaning representation) to the
**generated text**.

Mark `has_error` as:

- **YES** — if ANY slot–value pair is incorrect.
- **NO**  — if ALL slot–value pairs are correctly realised in the text.

If YES, list errors in `error_details` using the following notation:

| Notation | Meaning |
|---|---|
| `D:slot`               | **Deletion:** slot is in the MR but missing from the text. |
| `S:slot=wrong_val`     | **Substitution:** slot mentioned in text but with the wrong value. |
| `I:slot=hal_val`       | **Insertion:** slot in text but NOT in the MR (hallucinated). |

Multiple errors are comma-separated, e.g.

```
D:area, S:food=Japanese, I:priceRange=cheap
```

## Conventions

- **Variable placeholders** (`nameVariable`, `nearVariable`,
  `_area_variable_`, etc.) are always considered correct by convention
  — annotators were instructed not to mark these as errors.
- The E2E NLG Challenge slot set is the universe of valid slots:
  `name`, `eatType`, `food`, `priceRange`, `customerRating`, `area`,
  `familyFriendly`, `near`.
- `familyFriendly` value semantics: `yes` ⇔ the text affirms
  family-friendliness; `no` ⇔ the text affirms not-family-friendliness.
  A text with no opinion on the matter when the MR says `yes`/`no`
  counts as a **D**eletion (or **S**ubstitution if the text says the
  opposite).

## Workflow

1. Run `xdomain_ser.eval2.sample` to draw a stratified 1000-example
   sample from upstream M2T outputs. Output: `raw/sampled-examples.json`.
2. Run `xdomain_ser.eval2.create_files` to emit TSV +
   structured-text annotation files (with the instruction block above
   prepended to the text version).
3. Annotators fill in `has_error` and `error_details` for each row.
4. Run `xdomain_ser.eval2.process` to parse the annotated file into
   `gold-annotated.json` with structured error lists and computed
   `gold_ser` / `gold_pred_mr` fields.

## Inter-annotator agreement

Not measured in v1 — the gold labels shipped here are single-annotator
output from the paper's author. Future releases may add a secondary
annotator pass to produce IAA statistics.
