import os
import random
import re
from collections import Counter

GRAMMAR_SCALE_WORDS = int(os.getenv("GRAMMAR_SCALE_WORDS", "0"))

ARTICLES = ["a", "an", "the"]

KEYBOARD_TYPOS = {"a": "s", "s": "a", "e": "w", "w": "e", "i": "o", "o": "i",
                  "u": "y", "y": "u", "n": "m", "m": "n"}

CONFUSIONS = {
    "their": "there", "there": "their", "they're": "there",
    "your": "ur", "you're": "your", "its": "it's", "it's": "its",
    "then": "than", "than": "then", "to": "2", "too": "to", "for": "4",
    "are": "r", "you": "u", "why": "y", "and": "nd", "because": "coz", "please": "plz",
    "affect": "effect", "effect": "affect", "loose": "lose", "lose": "loose",
    "accept": "except", "except": "accept", "ok": "k", "okay": "k",
    "message": "msg", "brother": "bro",
    "hai": "h", "hain": "h", "mein": "m", "kya": "k", "ki": "k", "ke": "k", "ka": "k",
    "nahi": "ni", "nahin": "ni", "kuch": "kch", "kaise": "kse", "kaisa": "ksa",
    "kyun": "q", "kyu": "q", "kyunki": "cuz", "mujhe": "mje", "tujhe": "tje",
    "bahut": "bht", "bohot": "bht", "haan": "hn", "yaar": "yr", "bhi": "b",
    "hum": "hm", "tum": "tm", "theek": "tk", "thik": "tk", "matlab": "mtlb",
    "waise": "vse", "jaldi": "jldi", "bhai": "bhy", "bhaiya": "bhy",
    "pata": "pta", "kab": "kb", "sab": "sb",
}

PHONETIC_RULES = [("sh", "s", "phonetic"), ("v", "w", "phonetic"), ("ph", "f", "phonetic"),
                  ("z", "j", "phonetic"), ("bh", "b", "aspirate_drop"), ("dh", "d", "aspirate_drop"),
                  ("gh", "g", "aspirate_drop"), ("kh", "k", "aspirate_drop"), ("jh", "j", "aspirate_drop")]

MASC_NOUNS = {"ghar", "kaam", "phone", "mobile", "laptop", "khana", "paani", "school", "office",
              "bazaar", "din", "saal", "mahina", "paisa", "dost", "bhai", "beta", "papa", "ladka",
              "kamra", "sapna", "gaana", "khel", "rasta", "shehar", "desh", "bag", "project",
              "result", "exam", "plan", "match", "darwaza", "kapda", "doodh", "pati", "bachcha"}
FEM_NOUNS = {"kitab", "kitaab", "baat", "movie", "film", "gaadi", "chai", "raat", "subah", "shaam",
             "cheez", "class", "meeting", "party", "naukri", "job", "zindagi", "family", "mummy",
             "maa", "behen", "bahen", "ladki", "beti", "dawai", "chhutti", "padhai", "kahani",
             "duniya", "problem", "team", "company", "trip", "biwi", "tabiyat", "khushi", "salary"}
MOD_PAIRS = [("mera", "meri"), ("tera", "teri"), ("uska", "uski"), ("unka", "unki"), ("iska", "iski"),
             ("hamara", "hamari"), ("tumhara", "tumhari"), ("apna", "apni"), ("ka", "ki"),
             ("accha", "acchi"), ("acha", "achi"), ("achha", "achhi"), ("bada", "badi"),
             ("chhota", "chhoti"), ("naya", "nayi"), ("purana", "purani"), ("lamba", "lambi"),
             ("pura", "puri"), ("kaisa", "kaisi"), ("kitna", "kitni")]
MASC_TO_FEM = {m: f for m, f in MOD_PAIRS}
FEM_TO_MASC = {f: m for m, f in MOD_PAIRS}

HONORIFIC_SWAPS = {"rahe": "raha", "karte": "karta", "jaate": "jaata", "jate": "jata", "aate": "aata",
                   "rehte": "rehta", "sakte": "sakta", "chahte": "chahta", "gaye": "gaya",
                   "aaye": "aaya", "karenge": "karega", "jayenge": "jayega", "aayenge": "aayega",
                   "honge": "hoga", "lenge": "lega", "denge": "dega", "hain": "hai"}

NE_FORMS = {"main": "maine", "mai": "maine", "hum": "humne", "tum": "tumne",
            "wo": "usne", "woh": "usne"}
NE_TO_PLAIN = {"maine": "main", "humne": "hum", "tumne": "tum", "usne": "wo",
               "unhone": "wo", "aapne": "aap"}
INTRANSITIVE_PAST = {"gaya", "gayi", "gai", "gaye", "aaya", "aayi", "aaye", "chala", "chali", "chale",
                     "hua", "hui", "hue", "baitha", "baithi", "baithe", "soya", "soyi", "soye",
                     "pahuncha", "pahunchi", "pahunche", "ruka", "ruki", "ruke", "utha", "uthi", "uthe",
                     "nikla", "nikli", "nikle", "lauta", "lauti", "laute"}
TRANSITIVE_PAST = {"kiya", "kiye", "dekha", "dekhi", "dekhe", "khaya", "khayi", "khaye", "bola", "boli",
                   "suna", "suni", "padha", "padhi", "likha", "likhi", "liya", "diya", "socha",
                   "samjha", "bheja", "bheji", "banaya", "banayi", "kharida", "kharidi", "pucha",
                   "poocha", "bataya", "kaha"}

KE_HEADS = {"liye", "saath", "sath", "baad", "baare", "bare", "paas", "upar", "neeche", "niche",
            "andar", "bahar", "pehle", "kaaran", "karan", "alawa", "khilaaf", "khilaf", "beech",
            "bina", "zariye", "dauran"}
KI_HEADS = {"taraf", "tarah", "jagah", "wajah", "vajah"}

PAST_CUES = {"pichle", "pichhle", "pichla", "pichli", "beete"}
FUTURE_CUES = {"agle", "agli", "agla"}
HINDI_PAST_TO_PRESENT = {"tha": "hai", "thi": "hai", "the": "hain"}
HINDI_FUTURE_TO_PAST = {"hoga": "tha", "hogi": "thi", "honge": "the"}

ENG_SG_SUBJ = {"he", "she", "it"}
ENG_PL_SUBJ = {"they", "we", "you"}
ENG_SG_TO_PL = {"is": "are", "has": "have", "does": "do", "was": "were"}
ENG_PL_TO_SG = {"are": "is", "have": "has", "do": "does", "were": "was"}
ENG_I = {"am": "is", "have": "has"}
ENG_PAST_CUES = {"yesterday", "last", "ago"}
ENG_PAST_TO_BASE = {"went": "go", "came": "come", "saw": "see", "ate": "eat", "bought": "buy",
                    "met": "meet", "took": "take", "gave": "give", "made": "make", "got": "get",
                    "told": "tell", "left": "leave", "wrote": "write", "called": "call",
                    "visited": "visit", "watched": "watch", "played": "play", "finished": "finish"}

BOUNDARY = {"aur", "lekin", "magar", "kyunki", "toh", "jab", "but", "because", "coz", "and",
            "or", "ya", "so", "phir"}

TAG_CATEGORY = {
    "gender": "hinglish_grammar", "person": "hinglish_grammar", "number": "hinglish_grammar",
    "honorific": "hinglish_grammar", "ne_drop": "hinglish_grammar", "ne_misuse": "hinglish_grammar",
    "compound_postposition": "hinglish_grammar", "tense": "hinglish_grammar",
    "eng_agreement": "english_grammar", "eng_tense": "english_grammar",
    "article_drop": "english_grammar", "article_swap": "english_grammar",
    "slang_confusion": "spelling", "phonetic": "spelling", "aspirate_drop": "spelling",
    "vowel_drop": "spelling", "keyboard_typo": "spelling", "char_delete": "spelling",
    "elongation": "spelling", "plural_s": "spelling",
    "word_drop": "structure", "swap_adjacent": "structure", "dup_word": "structure",
    "merge_words": "structure",
    "punct_strip": "punct_case", "punct_space": "punct_case", "lowercase_first": "punct_case",
    "random_upper": "punct_case",
    "clean": "clean",
}


def _match_case(orig, new):
    if orig[:1].isupper() and new:
        return new[0].upper() + new[1:]
    return new


def surface_noise(text, error_prob=0.15):
    tags = []
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text, tags
    words = text.split()
    if len(words) < 3:
        return text, tags

    new_words = []
    i = 0
    while i < len(words):
        word = words[i]
        action = random.random()

        if action < error_prob * 0.05:
            tags.append("word_drop")
            i += 1
            continue
        elif action < error_prob * 0.15:
            if word.lower() in ARTICLES:
                if random.random() < 0.5:
                    tags.append("article_drop")
                else:
                    new_words.append(random.choice([a for a in ARTICLES if a != word.lower()]))
                    tags.append("article_swap")
                i += 1
                continue
        elif action < error_prob * 0.22 and i < len(words) - 1:
            new_words.append(words[i + 1])
            new_words.append(word)
            tags.append("swap_adjacent")
            i += 2
            continue
        elif action < error_prob * 0.28:
            new_words.append(word)
            new_words.append(word)
            tags.append("dup_word")
            i += 1
            continue
        elif action < error_prob * 0.45:
            clean_word = word.lower().strip(".,!?")
            if clean_word in CONFUSIONS:
                new_word = CONFUSIONS[clean_word]
                if word.istitle():
                    new_word = new_word.title()
                trail = word[len(word.rstrip(".,!?")):]
                new_words.append(new_word + trail)
                tags.append("slang_confusion")
            else:
                new_words.append(word)
            i += 1
            continue
        elif action < error_prob * 0.55:
            clean_word = word.strip(".,!?")
            if len(clean_word) > 4 and clean_word.isalpha():
                if clean_word.endswith("s"):
                    new_words.append(word.replace(clean_word, clean_word[:-1]))
                else:
                    new_words.append(word.replace(clean_word, clean_word + "s"))
                tags.append("plural_s")
            else:
                new_words.append(word)
            i += 1
            continue
        elif action < error_prob * 0.65:
            options = [(a, b, t) for a, b, t in PHONETIC_RULES if a in word.lower()]
            if options:
                a, b, t = random.choice(options)
                new_word = word.replace(a, b).replace(a.capitalize(), b.capitalize())
                if new_word != word:
                    tags.append(t)
                new_words.append(new_word)
            else:
                new_words.append(word)
            i += 1
            continue
        elif action < error_prob * 0.72:
            if len(word) > 4:
                vowels = "aeiouAEIOU"
                new_word = "".join(c for idx, c in enumerate(word)
                                   if idx == 0 or idx == len(word) - 1 or c not in vowels)
                if len(new_word) > 2 and new_word != word:
                    new_words.append(new_word)
                    tags.append("vowel_drop")
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
            i += 1
            continue
        elif action < error_prob * 0.80:
            if len(word) > 3:
                chars = list(word)
                idx = random.randint(0, len(chars) - 1)
                c = chars[idx].lower()
                if c in KEYBOARD_TYPOS:
                    chars[idx] = KEYBOARD_TYPOS[c] if chars[idx].islower() else KEYBOARD_TYPOS[c].upper()
                    tags.append("keyboard_typo")
                new_words.append("".join(chars))
            else:
                new_words.append(word)
            i += 1
            continue
        elif action < error_prob * 0.86:
            if len(word) > 4:
                char_idx = random.randint(0, len(word) - 1)
                word = word[:char_idx] + word[char_idx + 1:]
                tags.append("char_delete")
            new_words.append(word)
            i += 1
            continue
        elif action < error_prob * 0.94:
            if len(word) >= 2 and word.isalpha():
                chars = list(word)
                for v_idx, char in enumerate(chars):
                    if char in "aeiouAEIOU":
                        chars[v_idx] = char * random.randint(3, 5)
                        tags.append("elongation")
                        break
                new_words.append("".join(chars))
            else:
                new_words.append(word)
            i += 1
            continue
        elif action < error_prob * 1.0 and i < len(words) - 1:
            new_words.append(word + words[i + 1])
            tags.append("merge_words")
            i += 2
            continue
        else:
            new_words.append(word)
            i += 1

    result = " ".join(new_words)

    if random.random() < error_prob * 1.5:
        before = result
        punct_action = random.random()
        if punct_action < 0.25:
            result = result.replace(".", "").replace(",", "").replace("!", "?")
            tag = "punct_strip"
        elif punct_action < 0.50:
            result = re.sub(r"([.,!?])\s+", r"\1", result)
            tag = "punct_space"
        elif punct_action < 0.75:
            if result:
                result = result[0].lower() + result[1:]
            tag = "lowercase_first"
        else:
            if len(result) > 5:
                idx = random.randint(1, len(result) - 2)
                result = result[:idx] + result[idx].upper() + result[idx + 1:]
            tag = "random_upper"
        if result != before:
            tags.append(tag)

    if result.strip() == "":
        return text, []
    return result, tags


def _split_token(tok):
    return list(re.match(r"^([^\w]*)(.*?)([^\w]*)$", tok).groups())


def _has_break(part):
    return any(c in part[2] for c in ".?!;,")


def _words_before(parts, i, size=6):
    out = []
    for j in range(i - 1, max(-1, i - 1 - size), -1):
        if _has_break(parts[j]):
            break
        w = parts[j][1].lower()
        if w in BOUNDARY:
            break
        out.append(w)
    return out


def _words_after(parts, i, size=6):
    out = []
    if _has_break(parts[i]):
        return out
    for j in range(i + 1, min(len(parts), i + 1 + size)):
        w = parts[j][1].lower()
        if w in BOUNDARY:
            break
        out.append(w)
        if _has_break(parts[j]):
            break
    return out


def _clause_final(parts, i):
    if i == len(parts) - 1 or _has_break(parts[i]):
        return True
    return parts[i + 1][1].lower() in BOUNDARY | {"nahi", "kya", "isliye", "par", "lekin"}


def g_gender(parts, i, ctx):
    w = parts[i][1].lower()
    if i >= len(parts) - 1 or _has_break(parts[i]):
        return None
    nxt = parts[i + 1][1].lower()
    if w in MASC_TO_FEM and nxt in MASC_NOUNS:
        return MASC_TO_FEM[w]
    if w in FEM_TO_MASC and nxt in FEM_NOUNS:
        return FEM_TO_MASC[w]
    return None


def g_person(parts, i, ctx):
    w = parts[i][1].lower()
    back = set(_words_before(parts, i))
    if w in {"hoon", "hu", "hun"} and back & {"main", "mai"}:
        return random.choice(["hai", "ho"])
    if w == "ho" and "tum" in back and "aap" not in back:
        return random.choice(["hai", "hoon"])
    return None


def g_number(parts, i, ctx):
    w = parts[i][1].lower()
    back = set(_words_before(parts, i))
    if w == "hain" and back & {"hum", "log", "sab", "sabhi"} and "aap" not in back:
        return "hai"
    return None


def g_honorific(parts, i, ctx):
    w = parts[i][1].lower()
    if w not in HONORIFIC_SWAPS:
        return None
    back = set(_words_before(parts, i))
    if "aap" in back and not back & {"hum", "log"}:
        return HONORIFIC_SWAPS[w]
    return None


def g_ne_drop(parts, i, ctx):
    w = parts[i][1].lower()
    if w in NE_TO_PLAIN and set(_words_after(parts, i)) & TRANSITIVE_PAST:
        return NE_TO_PLAIN[w]
    return None


def g_ne_misuse(parts, i, ctx):
    w = parts[i][1].lower()
    if w not in NE_FORMS:
        return None
    after = set(_words_after(parts, i))
    if after & INTRANSITIVE_PAST and not after & TRANSITIVE_PAST:
        return NE_FORMS[w]
    return None


def g_compound_postposition(parts, i, ctx):
    if i >= len(parts) - 1 or _has_break(parts[i]):
        return None
    w, nxt = parts[i][1].lower(), parts[i + 1][1].lower()
    if w == "ke" and nxt in KE_HEADS:
        return random.choice(["ki", "ka"])
    if w == "ki" and nxt in KI_HEADS:
        return random.choice(["ke", "ka"])
    return None


def g_tense(parts, i, ctx):
    w = parts[i][1].lower()
    if ctx["hindi_past"] and w in HINDI_PAST_TO_PRESENT and _clause_final(parts, i):
        return HINDI_PAST_TO_PRESENT[w]
    if ctx["hindi_future"] and w in HINDI_FUTURE_TO_PAST and _clause_final(parts, i):
        return HINDI_FUTURE_TO_PAST[w]
    return None


def g_eng_agreement(parts, i, ctx):
    if i == 0:
        return None
    w, prev = parts[i][1].lower(), parts[i - 1][1].lower()
    if _has_break(parts[i - 1]):
        return None
    if prev in ENG_SG_SUBJ and w in ENG_SG_TO_PL:
        return ENG_SG_TO_PL[w]
    if prev in ENG_PL_SUBJ and w in ENG_PL_TO_SG:
        return ENG_PL_TO_SG[w]
    if prev == "i" and w in ENG_I:
        return ENG_I[w]
    return None


def g_eng_tense(parts, i, ctx):
    w = parts[i][1].lower()
    if ctx["english_past"] and w in ENG_PAST_TO_BASE:
        return ENG_PAST_TO_BASE[w]
    return None


GENERATORS = [
    ("gender", g_gender), ("person", g_person), ("number", g_number), ("honorific", g_honorific),
    ("ne_drop", g_ne_drop), ("ne_misuse", g_ne_misuse),
    ("compound_postposition", g_compound_postposition), ("tense", g_tense),
    ("eng_agreement", g_eng_agreement), ("eng_tense", g_eng_tense),
]


def grammar_noise(text, max_errors=2, pick_prob=0.5):
    if not isinstance(text, str) or len(text.split()) < 3:
        return text, []
    parts = [_split_token(t) for t in text.split()]
    words = {p[1].lower() for p in parts}
    ctx = {"hindi_past": bool(words & PAST_CUES), "hindi_future": bool(words & FUTURE_CUES),
           "english_past": bool(words & ENG_PAST_CUES)}

    candidates = []
    for i in range(len(parts)):
        if not parts[i][1]:
            continue
        for name, fn in GENERATORS:
            new = fn(parts, i, ctx)
            if new is not None and new != parts[i][1].lower():
                candidates.append((i, name, new))
    if not candidates:
        return text, []

    random.shuffle(candidates)
    chosen, used = [], set()
    for i, name, new in candidates:
        if len(chosen) >= max_errors:
            break
        if i not in used and random.random() < pick_prob:
            chosen.append((i, name, new))
            used.add(i)
    if not chosen:
        chosen = [candidates[0]]

    tags = []
    for i, name, new in chosen:
        parts[i][1] = _match_case(parts[i][1], new)
        tags.append(name)
    out = " ".join(a + b + c for a, b, c in parts)
    return re.sub(r"\s+", " ", out).strip(), tags


DENSITIES = {"light": (0.08, 1), "medium": (0.15, 2), "heavy": (0.30, 3)}
DENSITY_WEIGHTS = {"light": 0.30, "medium": 0.45, "heavy": 0.25}
MODE_WEIGHTS = {"surface": 0.35, "grammar": 0.25, "both": 0.40}


def noisy_message(text, p_clean=0.15):
    if not isinstance(text, str) or len(text.split()) < 3 or random.random() < p_clean:
        return text, ["clean"]
    density = random.choices(list(DENSITY_WEIGHTS), weights=list(DENSITY_WEIGHTS.values()))[0]
    surface_p, g_max = DENSITIES[density]
    if GRAMMAR_SCALE_WORDS > 0:
        g_max *= max(1, int(len(text.split()) / GRAMMAR_SCALE_WORDS + 0.5))
    mode = random.choices(list(MODE_WEIGHTS), weights=list(MODE_WEIGHTS.values()))[0]

    for attempt_mode, p in ((mode, surface_p), ("surface", surface_p), ("surface", 0.30)):
        out, tags = text, []
        if attempt_mode in ("grammar", "both"):
            out, g_tags = grammar_noise(out, max_errors=g_max)
            tags += g_tags
        if attempt_mode in ("surface", "both"):
            out, s_tags = surface_noise(out, p)
            tags += s_tags
        if out != text:
            return out, sorted(set(tags))
    return text, ["clean"]


def coverage_report(tag_strings, title):
    tag_lists = [str(t).split(";") if isinstance(t, str) and t else ["clean"] for t in tag_strings]
    counts = Counter(t for tl in tag_lists for t in set(tl))
    n = len(tag_lists)
    lines = [f"===== COVERAGE: {title} ({n:,} examples) ====="]
    for cat in ["hinglish_grammar", "english_grammar", "spelling", "structure", "punct_case", "clean"]:
        cat_tags = [t for t, c in TAG_CATEGORY.items() if c == cat]
        in_cat = sum(1 for tl in tag_lists if any(TAG_CATEGORY.get(t) == cat for t in tl))
        lines.append(f"[{cat}] examples with at least one: {in_cat:,} ({100 * in_cat / max(n, 1):.1f}%)")
        for t in cat_tags:
            c = counts.get(t, 0)
            lines.append(f"    {t:<24}{c:>9,}  ({100 * c / max(n, 1):5.1f}%)")
    return lines
