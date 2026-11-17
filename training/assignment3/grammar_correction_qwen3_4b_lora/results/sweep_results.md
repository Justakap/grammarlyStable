# Learning rate and LoRA rank sweep (validation sets)

Run D's setup (Assignment 2 adapter trained further on Data Mix D), each configuration trained on a random 50,000
pairs and scored on the validation sets (1,500 IndicTalk and 500 Assignment 2 pairs). Made by
`code/sweep_lr_rank/select_best.py`.

| run | lr | rank | f05_indictalk | f05_a2 | mean_f05 | mean_over_correction | mean_gleu | mean_exact_match |
|---|---|---|---|---|---|---|---|---|
| lr1e-4_r16 | 1e-4 | 16 | 90.79 | 91.33 | 91.06 | 2.87 | 96.55 | 73.03 |
| lr1e-4_r32 | 1e-4 | 32 | 91.14 | 90.95 | 91.04 | 2.87 | 96.61 | 73.40 |
| lr2e-4_r32 | 2e-4 | 32 | 91.33 | 90.75 | 91.04 | 1.91 | 96.59 | 73.80 |
| lr5e-5_r32 | 5e-5 | 32 | 90.75 | 91.14 | 90.95 | 3.82 | 96.55 | 72.90 |
| lr1e-4_r64 | 1e-4 | 64 | 91.11 | 90.67 | 90.89 | 2.61 | 96.56 | 73.20 |

Rule: highest mean edit F0.5 over val_indictalk and val_a2; runs within 1.0 point of the best count as tied, and
among those the lowest over-correction wins, then the smaller rank.

All five configurations are tied (spread 0.17 points). **Selected: lr2e-4_r32** (learning rate 2e-4, rank 32,
mean F0.5 91.04, over-correction 1.91%). In this range neither the learning rate nor the rank matters; rank 16 is
as good as 64.
