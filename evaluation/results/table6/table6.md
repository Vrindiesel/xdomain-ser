# Table 6 reproduction — vs rule-based SER tools (six topics)

Values vs published (|delta| in parentheses); paper rows only.

| topic | row | S | D | I | All | MAE |
|---|---|---|---|---|---|---|
| restaurants | E2E script | 0.881 (Δ0.000) | 0.730 (Δ0.000) | 0.940 (Δ0.000) | 0.699 (Δ0.000) | 0.046 (Δ0.000) |
| restaurants | LR-Routing | 0.836 (Δ0.000) | 0.753 (Δ0.001) | 0.942 (Δ0.000) | 0.721 (Δ0.000) | 0.043 (Δ0.000) |
| restaurant | RNNLG | 0.971 (Δ0.001) | 0.876 (Δ0.000) | 0.957 (Δ0.000) | 0.827 (Δ0.000) | 0.055 (Δ0.000) |
| restaurant | LR-Routing | 0.989 (Δ0.000) | 1.000 (Δ0.000) | 1.000 (Δ0.000) | 0.989 (Δ0.000) | 0.003 (Δ0.000) |
| hotel | RNNLG | 0.902 (Δ0.000) | 0.658 (Δ0.000) | 0.823 (Δ0.000) | 0.504 (Δ0.000) | 0.193 (Δ0.000) |
| hotel | LR-Routing | 0.973 (Δ0.000) | 0.979 (Δ0.000) | 0.985 (Δ0.000) | 0.944 (Δ0.000) | 0.020 (Δ0.000) |
| laptop | RNNLG | 0.873 (Δ0.000) | 0.676 (Δ0.000) | 0.881 (Δ0.000) | 0.544 (Δ0.000) | 0.117 (Δ0.000) |
| laptop | NLI | 0.951 (Δ0.000) | 0.917 (Δ0.000) | 0.960 (Δ0.000) | 0.870 (Δ0.000) | 0.024 (Δ0.000) |
| tv | RNNLG | 0.764 (Δ0.000) | 0.538 (Δ0.000) | 0.838 (Δ0.000) | 0.336 (Δ0.000) | 0.174 (Δ0.000) |
| tv | ScoreRouting | 0.927 (Δ0.000) | 0.870 (Δ0.000) | 0.943 (Δ0.000) | 0.788 (Δ0.000) | 0.048 (Δ0.000) |
| video_games | ViGGO | 0.774 (Δ0.000) | 0.916 (Δ0.000) | 0.983 (Δ0.000) | 0.720 (Δ0.000) | 0.071 (Δ0.000) |
| video_games | LR-Routing | 0.892 (Δ0.000) | 0.868 (Δ0.000) | 0.934 (Δ0.000) | 0.753 (Δ0.000) | 0.071 (Δ0.000) |

## Protocol notes

1. Aligner rows are per-text, true-MR-inventory conditioned (P-oracle), as published; not protocol-invariant under P-deploy.
2. As published, aligner rows use ALL pairs of the topic; learned rows use the seed-42 test half. `basis=test` aligner rows in table6.tsv are the like-for-like extra (not in the paper).
3. RNNLG rows replicate the published default-domain quirk (all four topics extracted with domain='restaurant').
