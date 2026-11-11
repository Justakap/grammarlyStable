# Grammar correction results on long messages

Long-message IndicTalk test set: 2,000 whole messages of up to 120 words (`data_a3_full/test_indictalk.csv`).

## test_indictalk (long messages)

| run | n | exact_match | over_correction_rate | chrf | chrf_pp | gleu | edit_f0_5 | bleu |
|---|---|---|---|---|---|---|---|---|
| G2 | 2000 | 64.10 | 1.96 | 99.14 | 99.09 | 97.31 | 91.16 | 98.15 |
| G2_long | 2000 | 67.55 | 1.63 | 99.28 | 99.25 | 97.77 | 92.89 | 98.48 |
| 8B_all | 2000 | 67.90 | 0.98 | 99.29 | 99.26 | 97.85 | 93.15 | 98.53 |
| copy input | 2000 | 15.30 | 0.00 | 95.70 | 94.51 | 70.25 | 0.00 | 86.67 |
