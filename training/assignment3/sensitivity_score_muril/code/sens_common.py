import re
import math

# Sensitivity score: 0 (safe) to 100 (offensive or hateful), from one sigmoid output.
# Targets: TRAC NAG (not aggressive) 0.0, CAG (covert aggression) CAG_TARGET, OAG (overt aggression) 1.0;
# Kaggle-style hate labels 0 / 1; safe chat (IndicTalk, Assignment 2 without profanity) 0.0.
TRAC_LABELS = ("NAG", "CAG", "OAG")
BANDS = ((30.0, "safe"), (70.0, "sensitive"), (101.0, "offensive"))

INVISIBLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏  ﻿]")
URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
MENTION = re.compile(r"@\w+")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
WORD = re.compile(r"[a-z]+")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = INVISIBLE.sub(" ", text)
    text = URL.sub(" ", text)
    text = MENTION.sub("@user", text)
    return re.sub(r"\s+", " ", text).strip()


def is_romanized(text, max_devanagari_share=0.1, min_latin=3):
    dev = len(DEVANAGARI.findall(text))
    lat = len(LATIN.findall(text))
    return lat >= min_latin and dev <= max_devanagari_share * (dev + lat)


def words(text):
    return WORD.findall(text.lower())


def load_lexicon(path, min_severity=1):
    """HOT Hinglish_Profanity_List.csv: word,english,severity (1-10)."""
    out = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if not parts or not parts[0]:
                continue
            try:
                sev = int(parts[-1])
            except ValueError:
                sev = min_severity
            if sev >= min_severity:
                out.add(parts[0].lower())
    return out


def has_lexicon_word(text, lexicon):
    return any(w in lexicon for w in words(text))


def band(score):
    for limit, name in BANDS:
        if score < limit:
            return name
    return BANDS[-1][1]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def auroc(labels, scores):
    """labels 0/1. Rank-sum AUROC with average ranks for ties. None if one class is missing."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(l for _, l in pairs[i:j + 1])
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ece(labels, probs, bins=10):
    """Expected calibration error for binary labels and probabilities in [0, 1]."""
    if not labels:
        return None
    total, err = len(labels), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [k for k, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[k] for k in idx) / len(idx)
            acc = sum(labels[k] for k in idx) / len(idx)
            err += len(idx) / total * abs(acc - conf)
    return err


def prf(labels, preds):
    tp = sum(1 for l, p in zip(labels, preds) if l and p)
    fp = sum(1 for l, p in zip(labels, preds) if not l and p)
    fn = sum(1 for l, p in zip(labels, preds) if l and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def macro_f1(gold, pred, classes):
    f1s = []
    for c in classes:
        _, _, f1 = prf([g == c for g in gold], [p == c for p in pred])
        f1s.append(f1)
    return sum(f1s) / len(f1s)
