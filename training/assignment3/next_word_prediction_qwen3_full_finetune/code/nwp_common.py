import re

PERSONAS = ["friends", "family", "colleagues", "experts", "student_teacher"]
WRONG_PERSONA = {"friends": "experts", "family": "colleagues", "colleagues": "friends",
                 "experts": "student_teacher", "student_teacher": "family", "none": "none"}
PERSONA_MODES = ("true", "none", "wrong", "off")

INVISIBLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​‎‏  ﻿]")
WORD = re.compile(r"\S+")
EDGE_PUNCT = "\"'`.,!?;:()[]{}<>*_~|-–—…“”‘’"


def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", INVISIBLE.sub("", text)).strip()


def clean_persona(persona):
    p = str(persona or "none").strip().lower().replace(" ", "_").replace("-", "_")
    return p if p in PERSONAS else "none"


def persona_for_mode(persona, mode):
    persona = clean_persona(persona)
    if mode == "true":
        return persona
    if mode == "none":
        return "none"
    if mode == "wrong":
        return WRONG_PERSONA[persona]
    return None


def persona_prefix(persona):
    return "" if persona is None else f"Persona: {persona}\n"


def separator(name):
    return " " if name == "space" else "\n"


def join_turns(turns, sep_name="newline"):
    return separator(sep_name).join(turns)


def chunk_turns(turns, max_words):
    chunks, current, count = [], [], 0
    for turn in turns:
        n = len(turn.split())
        if current and count + n > max_words:
            chunks.append(current)
            current, count = [], 0
        if n > max_words:
            words = turn.split()
            for s in range(0, len(words), max_words):
                chunks.append([" ".join(words[s:s + max_words])])
            continue
        current.append(turn)
        count += n
    if current:
        chunks.append(current)
    return chunks


def norm_word(word):
    return word.strip().strip(EDGE_PUNCT).lower()


def has_alnum(word):
    return any(c.isalnum() for c in word)


def word_positions(turns, sep_name="newline"):
    sep = separator(sep_name)
    out, offset = [], 0
    for t, turn in enumerate(turns):
        spans = [(m.start(), m.end(), m.group()) for m in WORD.finditer(turn)]
        for j in range(1, len(spans)):
            target = norm_word(spans[j][2])
            if target and has_alnum(target):
                out.append((offset + spans[j - 1][1], target))
        offset += len(turn) + len(sep)
    return out


def is_word_start(piece):
    return piece.startswith(" ") and has_alnum(piece)


def build_word(pieces):
    word = pieces[0]
    for piece in pieces[1:]:
        if piece == "" or piece[0].isspace():
            break
        word += piece
    return norm_word(word)
