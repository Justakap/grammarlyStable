# Next-word prediction results

Token-level: perplexity and next-token top-1/top-3 accuracy (%). Word-level: the next whole word is among the model's suggestions (%), mean reciprocal rank, and the share of characters saved by accepting a correct suggestion at the start of each word (%).

## test_indictalk

| run | persona_mode | perplexity | top1 | top3 | word_top1 | word_top3 | word_mrr | keystroke_savings | tokens | words |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.6B | none | 6.16 | 58.06 | 75.68 | 41.00 | 60.40 | 0.495 | 53.58 | 1589617 | 3000 |
| 0.6B | true | 5.92 | 58.60 | 76.23 | 41.53 | 61.17 | 0.501 | 54.35 | 1589617 | 3000 |
| 0.6B | wrong | 6.19 | 57.90 | 75.55 | 41.20 | 59.67 | 0.494 | 52.84 | 1589617 | 3000 |
| 1.7B | none | 4.94 | 61.09 | 78.92 | 46.17 | 64.50 | 0.543 | 58.56 | 1589617 | 3000 |
| 1.7B | true | 4.76 | 61.60 | 79.49 | 46.30 | 65.67 | 0.548 | 59.60 | 1589617 | 3000 |
| 1.7B | wrong | 5.01 | 60.79 | 78.71 | 45.83 | 64.63 | 0.541 | 58.59 | 1589617 | 3000 |
| A2 | off | 31.70 | 37.07 | 53.09 | 21.10 | 32.63 | 0.261 | 26.74 | 1577733 | 3000 |
| BASE | off | 54.76 | 31.70 | 47.27 | 16.17 | 26.87 | 0.208 | 21.52 | 1586288 | 3000 |

## test_a2

| run | persona_mode | perplexity | top1 | top3 | word_top1 | word_top3 | word_mrr | keystroke_savings | tokens | words |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.6B | true | 5.20 | 60.55 | 78.16 | 46.87 | 65.17 | 0.551 | 61.03 | 5114744 | 3000 |
| 1.7B | true | 4.18 | 64.14 | 81.50 | 50.47 | 68.70 | 0.587 | 65.10 | 5114744 | 3000 |
| A2 | off | 15.03 | 45.91 | 62.65 | 30.83 | 46.30 | 0.377 | 42.40 | 5025442 | 3000 |
| BASE | off | 46.72 | 33.16 | 48.58 | 21.10 | 32.80 | 0.262 | 29.34 | 5073915 | 3000 |
