# Sensitivity score results

Score = 100 x calibrated sigmoid. Bands: under 30 safe, 30-70 sensitive, 70 and over offensive.

## TRAC-1 Hindi test (Romanized Facebook comments, official dev set)

| run | n | AUROC NAG vs OAG | AUROC NAG vs aggressive | mean NAG / CAG / OAG | band acc | band macro-F1 | OAG F1 @50 | ECE before / after |
|---|---|---|---|---|---|---|---|---|
| muril | 2160 | 87.73 | 82.00 | 34.1 / 57.82 / 71.53 | 59.68 | 57.55 | 66.82 | 13.80 / 13.80 |
| qwen0.6b | 2160 | 87.18 | 81.88 | 38.84 / 56.29 / 69.13 | 58.47 | 54.07 | 65.66 | 13.72 / 15.11 |

## Everyday chat (false alarms) and profanity check

| run | test | n | mean score | share >= 30 (%) | share >= 70 (%) |
|---|---|---|---|---|---|
| muril | test_safe_indictalk | 1000 | 1.56 | 0.0 | 0.0 |
| muril | test_safe_a2 | 1000 | 1.8 | 0.3 | 0.0 |
| muril | test_a2_lexicon | 81 | 2.25 | 1.23 | 0.0 |
| qwen0.6b | test_safe_indictalk | 1000 | 0.08 | 0.0 | 0.0 |
| qwen0.6b | test_safe_a2 | 1000 | 0.65 | 0.4 | 0.0 |
| qwen0.6b | test_a2_lexicon | 81 | 5.45 | 6.17 | 1.23 |

For the safe-chat sets lower is better (false alarms); for test_a2_lexicon (messages with a strong profanity word) higher is better.

## Sample messages

| run | score | band | text |
|---|---|---|---|
| muril | 11.4 | safe | kal milte hai yaar, movie ka plan pakka |
| muril | 2.8 | safe | bhai tu bahut accha kaam kar raha hai, keep it up |
| muril | 57.6 | sensitive | ye government kuch nahi karti, sab chor hai |
| muril | 81.2 | offensive | tu pagal hai kya, dimag kharab hai tera |
| muril | 87.2 | offensive | chup kar bewakoof, teri aukaat kya hai |
| muril | 71.1 | offensive | in logo ko desh se bahar nikal do, ye sab gaddar hai |
| qwen0.6b | 30.6 | sensitive | kal milte hai yaar, movie ka plan pakka |
| qwen0.6b | 1.2 | safe | bhai tu bahut accha kaam kar raha hai, keep it up |
| qwen0.6b | 50.4 | sensitive | ye government kuch nahi karti, sab chor hai |
| qwen0.6b | 71.7 | offensive | tu pagal hai kya, dimag kharab hai tera |
| qwen0.6b | 83.3 | offensive | chup kar bewakoof, teri aukaat kya hai |
| qwen0.6b | 42.1 | sensitive | in logo ko desh se bahar nikal do, ye sab gaddar hai |
