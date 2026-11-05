"""RFQ parser for a B2B packaging marketplace (Kypson pre-interview task).

    parse_rfq(text) -> {"category", "quantity", "unit", "gsm", "micron", "location"}

Approach: hybrid, rules-first, standard library only (no model, no network).
Every field has its own small, explainable extractor, so a wrong answer can be
traced to one rule.  Pipeline:

    raw text
      -> _normalize()      NFKC, lowercase, Devanagari digits/words, punctuation
                           noise, Indian digit grouping, "20k"/"2.5lac" spacing,
                           "g/m2" -> gsm, glued number+unit splitting
      -> _fix_typos()      explicit typo map + conservative fuzzy match against
                           a small keyword vocabulary (difflib, stdlib)
      -> _words_to_digits() "do hazaar" -> "2 hazaar", "twenty five thousand"
      -> _extract_specs()  GSM and micron (also yields spans to mask)
      -> _classify()       weighted keyword/phrase scoring per category
      -> _extract_quantity() candidate scoring (pcs vs weight vs capacity)
      -> _extract_location() gazetteer + fuzzy + grammatical-cue fallback
                           (works on the original-case text)

Run `python parse.py "your message"` to see the result and a short trace.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache

CATEGORIES = (
    "bopp_bag", "woven_sack", "corrugated_box", "pouch", "film",
    "mono_carton", "fibc",
)
_DEFAULT_CATEGORY = "woven_sack"  # most common generic "bag" when nothing else fires

# --------------------------------------------------------------------------
# 1. Normalisation
# --------------------------------------------------------------------------
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_DEVA_WORDS = {
    "हज़ार": "hazaar", "हजार": "hazaar", "लाख": "lakh", "करोड़": "crore",
    "करोड": "crore", "पीस": "pcs", "किलो": "kg", "टन": "ton",
}

# Exact-token typo / spelling-variant map (applied before fuzzy matching).
_TYPOS = {
    # Indian number words -> canonical multipliers
    "hazar": "hazaar", "hazzar": "hazaar", "hajar": "hazaar", "hajaar": "hazaar",
    "haazar": "hazaar", "hazaaar": "hazaar", "hzr": "hazaar", "hazr": "hazaar",
    "lac": "lakh", "lacs": "lakh", "lakhs": "lakh", "lak": "lakh", "laakh": "lakh",
    "lakh's": "lakh", "lakhh": "lakh", "crores": "crore", "thousands": "thousand",
    # units
    "pieces": "pcs", "piece": "pcs", "pc": "pcs", "pce": "pcs", "pices": "pcs",
    "peices": "pcs", "pcss": "pcs", "pics": "pcs", "nos": "pcs", "numbers": "pcs",
    "kilos": "kg", "kilo": "kg", "kgs": "kg", "kilogram": "kg", "kilograms": "kg",
    "mts": "mt", "tons": "ton", "tonnes": "ton", "tonne": "ton", "tonn": "ton",
    "tn": "ton", "tonns": "ton", "qtl": "quintal", "quintals": "quintal",
    # spec words
    "microns": "micron", "mikron": "micron", "mircon": "micron", "micorn": "micron",
    "micro": "micron", "maicron": "micron", "gsn": "gsm", "gmsm": "gsm",
    # product words
    "cartoon": "carton", "cartoons": "cartons", "carten": "carton", "cartan": "carton",
    "boxs": "boxes", "bagz": "bags", "begs": "bags", "bgs": "bags",
    "bop": "bopp", "bpp": "bopp", "boop": "bopp", "boppp": "bopp", "bopps": "bopp",
    "polybag": "poly bag", "polybags": "poly bags", "adstar": "ad star",
    "jumbbo": "jumbo", "jmbo": "jumbo", "jumbu": "jumbo", "jambo": "jumbo",
    "stratch": "stretch", "strech": "stretch", "streatch": "stretch",
    "shrinc": "shrink", "srink": "shrink", "flim": "film", "fillm": "film",
    "pouche": "pouch", "pouh": "pouch", "pauch": "pouch", "pouches": "pouch",
    "corrogated": "corrugated", "corugated": "corrugated", "corrugted": "corrugated",
    "corrugeted": "corrugated", "coragated": "corrugated",
    "monocarton": "mono carton", "monocartons": "mono cartons",
    "chahiye": "chahiye", "chaiye": "chahiye", "chahie": "chahiye",
    "chahiyen": "chahiye", "cahiye": "chahiye", "chahiyee": "chahiye",
    "qnty": "qty", "qty.": "qty", "quanity": "quantity", "quantiy": "quantity",
}

# Vocabulary for conservative fuzzy correction of misspelt keywords.
_FUZZY_TARGETS = (
    "jumbo", "corrugated", "carton", "cartons", "laminated", "woven", "shrink",
    "stretch", "pouch", "cement", "micron", "capacity", "sachet", "sacks",
    "boxes", "wrapping", "packaging", "quantity", "delivery", "required",
)
_FUZZY_BY_LETTER: dict[str, list[str]] = {}
for _w in _FUZZY_TARGETS:
    _FUZZY_BY_LETTER.setdefault(_w[0], []).append(_w)


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_DEVA_DIGITS)
    for k, v in _DEVA_WORDS.items():
        t = t.replace(k, f" {v} ")
    t = t.lower()
    t = re.sub(r"[‐-―−]", "-", t)
    t = t.replace("×", "x").replace("✕", "x").replace("✖", "x")
    t = t.replace("₹", " rs ").replace("μ", "μ")
    t = re.sub(r"(\d)\s*[\"”″]", r"\1 inch ", t)            # 12" -> 12 inch
    t = re.sub(r"\bg\s*\.\s*s\s*\.\s*m\b\.?", " gsm ", t)     # g.s.m
    t = re.sub(r"\bg(?:ms?|rams?)?\s*/\s*(?:m2|sqm|sq\.?\s*m|m)\b", " gsm ", t)  # g/m2
    t = re.sub(r"(\d)\s*/\s*-", r"\1 rs ", t)                # 5/- (price)
    t = re.sub(r"(?<=\d)\s*[x*]\s*(?=\d)", " x ", t)         # 10x12, 12*18
    # Indian / western digit grouping: 1,50,000  10,000  (comma between digits)
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r"(?<=\d),(?=\d{2,3}(?!\d))", "", t)
    # decimal comma before a multiplier: "2,5 lac"
    t = re.sub(r"(\d),(\d)(?=\s*(?:lakh|lac|lacs|lakhs|hazaar|hazar|k|crore|cr)\b)", r"\1.\2", t)
    t = re.sub(r"[^a-z0-9\s.,/%+@\-μ]", " ", t)              # punctuation noise
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)                # stray dots
    t = re.sub(r"(?<!\d)-|-(?!\d)", " ", t)                  # hyphen unless in range
    t = re.sub(r"(?<!\d)/|/(?!\d)", " ", t)
    t = t.replace("+", " ").replace("@", " @ ")
    t = re.sub(r"(?<=\d)(?=[a-zμ])", " ", t)                 # 40000pcs -> 40000 pcs
    t = re.sub(r"\s*,\s*", " , ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@lru_cache(maxsize=4096)
def _fix_token(tok: str) -> str:
    if tok in _TYPOS:
        return _TYPOS[tok]
    if len(tok) < 5 or tok in _KNOWN:
        return tok
    cands = [w for w in _FUZZY_BY_LETTER.get(tok[0], ()) if abs(len(w) - len(tok)) <= 2]
    best = get_close_matches(tok, cands, n=1, cutoff=0.82)
    return best[0] if best else tok


def _fix_typos(norm: str) -> str:
    return re.sub(r"[a-z]+", lambda m: _fix_token(m.group()), norm)


# --------------------------------------------------------------------------
# 2. Indian number words ("do hazaar", "dedh lakh", "twenty five thousand")
# --------------------------------------------------------------------------
_WORD_VALUES = {
    "ek": 1, "one": 1, "do": 2, "two": 2, "teen": 3, "three": 3, "char": 4,
    "chaar": 4, "four": 4, "panch": 5, "paanch": 5, "five": 5, "chhe": 6,
    "chhah": 6, "che": 6, "six": 6, "saat": 7, "seven": 7, "aath": 8, "eight": 8,
    "nau": 9, "nine": 9, "das": 10, "ten": 10, "gyarah": 11, "eleven": 11,
    "barah": 12, "twelve": 12, "terah": 13, "chaudah": 14, "pandrah": 15,
    "fifteen": 15, "solah": 16, "satrah": 17, "atharah": 18, "unnis": 19,
    "bees": 20, "twenty": 20, "pachis": 25, "pachchis": 25, "tees": 30,
    "thirty": 30, "chalis": 40, "forty": 40, "pachas": 50, "fifty": 50,
    "saath": 60, "sixty": 60, "sattar": 70, "seventy": 70, "assi": 80,
    "eighty": 80, "nabbe": 90, "ninety": 90,
    "dedh": 1.5, "dhai": 2.5, "dhaai": 2.5, "sava": 1.25, "sawa": 1.25,
    "aadha": 0.5, "adha": 0.5, "half": 0.5,
}
_WORD_MULTS = {"hazaar", "thousand", "lakh", "crore", "hundred", "sau"}
_WORD_SEQ_RE = re.compile(
    r"\b((?:(?:%s)\s+){1,3})(%s)\b" % ("|".join(sorted(_WORD_VALUES, key=len, reverse=True)),
                                     "|".join(_WORD_MULTS))
)


_COMPOUND_RE = re.compile(
    r"(?<![a-z\d.])(\d+(?:\.\d+)?)\s*(crore|lakh)\s+(\d+(?:\.\d+)?)\s*(lakh|hazaar|thousand|k)(?![a-z])")


def _merge_compound(norm: str) -> str:
    rank = {"crore": 3, "lakh": 2, "hazaar": 1, "thousand": 1, "k": 1}
    mult = {"crore": 10_000_000, "lakh": 100_000, "hazaar": 1_000, "thousand": 1_000, "k": 1_000}

    def repl(m: re.Match) -> str:
        a, ua, b, ub = float(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
        if rank[ub] >= rank[ua]:
            return m.group(0)
        total = a * mult[ua] + b * mult[ub]
        return str(int(total)) if total.is_integer() else str(total)
    return _COMPOUND_RE.sub(repl, norm)


def _words_to_digits(norm: str) -> str:
    def repl(m: re.Match) -> str:
        total = sum(_WORD_VALUES[w] for w in m.group(1).split())
        total = int(total) if float(total).is_integer() else total
        return f"{total} {m.group(2)}"
    return _merge_compound(_WORD_SEQ_RE.sub(repl, norm))


_MULT = {
    "k": 1_000, "thousand": 1_000, "hazaar": 1_000, "lakh": 100_000,
    "crore": 10_000_000, "cr": 10_000_000, "million": 1_000_000, "mn": 1_000_000,
    "hundred": 100, "sau": 100,
}
_MULT_ALT = "crore|lakh|hazaar|thousand|million|hundred|sau|cr|mn|k"
_NUM_RE = re.compile(
    r"(?<![a-z\d.])(?P<n1>\d+(?:\.\d+)?)(?:\s*(?P<m1>%(m)s)(?![a-z]))?"
    r"(?:\s*(?:-|to)\s*(?P<n2>\d+(?:\.\d+)?)(?:\s*(?P<m2>%(m)s)(?![a-z]))?)?" % {"m": _MULT_ALT}
)
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[a-z]+|,")


def _next_tokens(s: str, pos: int, n: int = 5) -> list[str]:
    out: list[str] = []
    for m in _TOKEN_RE.finditer(s, pos):
        out.append(m.group())
        if len(out) >= n:
            break
    return out


def _prev_tokens(s: str, pos: int, n: int = 4) -> list[str]:
    start = max(0, pos - 50)
    return _TOKEN_RE.findall(s[start:pos])[-n:] if start == 0 else _TOKEN_RE.findall(s[start:pos])[1:][-n:]


# --------------------------------------------------------------------------
# 3. GSM / micron / other numeric specs  (also produce spans to mask)
# --------------------------------------------------------------------------
_RANGE = r"(?:\s*(?:-|to|/)\s*\d{1,4}(?:\.\d+)?)?"
_GSM_NUM_RE = re.compile(r"(?<![\d.])(\d{1,4}(?:\.\d+)?)" + _RANGE + r"\s*gsm(?![a-z])")
_GSM_PRE_RE = re.compile(
    r"\bgsm\s*[:=]?\s*(\d{1,4}(?:\.\d+)?)(?![\d.])(?!\s*(?:micron|mic|mm|cm|pcs|kg|mt|ton|nos|bags?|k\b|lakh|hazaar))")
_GMS_RE = re.compile(r"(?<![\d.])(\d{2,4})\s*gms(?![a-z])\s*(\w*)")  # "gms" is a very common typo of gsm
_MIC_UNIT = r"(?:micron\w*|mics?|mikron\w*|micorn\w*|mircon\w*|maicron\w*|mu|um|μm?)(?![a-z])"
_MIC_NUM_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)" + _RANGE + r"\s*" + _MIC_UNIT)
_MIC_PRE_RE = re.compile(
    r"\b(?:micron\w*|mics?|thickness|thk|thick)\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)(?![\d.])"
    r"(?!\s*(?:mm|cm|gsm|gm|inch|kg|mt|ton|pcs|nos|bags?|k\b|lakh|hazaar|x\b))")

_UNIT_NOISE_RE = re.compile(
    r"(?<![a-z\d.])\d+(?:\.\d+)?\s*(?:inch|inches|mm|cm|ft|feet|foot|mtr|mtrs|meter|meters|metre|"
    r"metres|ml|ltr|ltrs|litre|litres|liter|liters|lt|gm|gram|grams|oz|ply|layer|layers|"
    r"layered|colour|colours|color|colors|side|sided|day|days|week|weeks|month|months|year|"
    r"years|hrs|hr|hour|hours|%|percent|rs|inr|rupee|rupees|usd|dollar|dollars|denier|mesh|"
    r"loop|loops|panel|panels|handle|handles|cut|wall|flute|pantone)(?![a-z])")
_PRICE_RE = re.compile(
    r"(?:\b(?:rs|inr|usd|price|rate|budget|target|cost|rupees|rupee)\b|@)\s*(?:rs|inr)?\s*\d+(?:\.\d+)?"
    r"(?:\s*(?:k|lakh|hazaar|thousand)(?![a-z]))?")
_PER_PRICE_RE = re.compile(r"(?<![a-z\d.])\d+(?:\.\d+)?\s*(?:per|/)\s*(?:pc|pcs|piece|kg|bag|bags|unit|roll|mt|ton|sqm|sq)(?![a-z])")
_PHONE_RE = re.compile(r"(?<![\d.])(?:\+?91\s?)?[6-9]\d{9}(?![\d.])|(?<![\d.])\d{11,}(?![\d.])")
_PIN_RE = re.compile(r"\bpin\s*(?:code)?\s*\d{6}(?!\d)")
_DIM_RE = re.compile(
    r"(?<![a-z\d.])\d+(?:\.\d+)?(?:\s*x\s*\d+(?:\.\d+)?)+"
    r"(?:\s*(?P<u>inch|inches|mm|cm|ft|feet|mtr|mtrs|meter|meters|metre|metres|m)(?![a-z]))?")

_COUNT_UNITS = {
    "pcs", "unit", "units", "bag", "bags", "sack", "sacks", "box", "boxes", "carton",
    "cartons", "ctn", "ctns", "pouch", "roll", "rolls", "set", "sets", "packet",
    "packets", "pkt", "pkts", "bundle", "bundles", "bori", "sheet", "sheets", "no",
    "number", "liner", "liners", "bale", "bales",
}


def _extract_specs(norm: str, category: str):
    """Return (gsm, micron, masked_spans). Spans hide spec numbers from the quantity step."""
    spans: list[tuple[int, int]] = []
    gsm = micron = None

    def to_int(s: str) -> int:
        return int(round(float(s)))

    for m in _GSM_NUM_RE.finditer(norm):
        if gsm is None:
            gsm = to_int(m.group(1))
        spans.append(m.span())
    if gsm is None:
        for m in _GSM_PRE_RE.finditer(norm):
            gsm = to_int(m.group(1))
            spans.append(m.span())
            break
    if gsm is None and category != "pouch":  # "70 gms" almost always means gsm unless it is a fill weight
        for m in _GMS_RE.finditer(norm):
            if m.group(2) not in {"pouch", "pack", "packet", "sachet", "bag", "box", "carton", "bags"}:
                gsm = to_int(m.group(1))
                spans.append((m.start(), m.start() + len(m.group(0)) - len(m.group(2))))
                break

    for m in _MIC_NUM_RE.finditer(norm):
        if micron is None:
            micron = to_int(m.group(1))
        spans.append(m.span())
    if micron is None:
        for m in _MIC_PRE_RE.finditer(norm):
            micron = to_int(m.group(1))
            spans.append(m.span())
            break

    if gsm is not None and not (5 <= gsm <= 2000):
        gsm = None
    if micron is not None and not (1 <= micron <= 1000):
        micron = None

    for rx in (_UNIT_NOISE_RE, _PRICE_RE, _PER_PRICE_RE, _PHONE_RE, _PIN_RE):
        spans.extend(m.span() for m in rx.finditer(norm))
    for m in _DIM_RE.finditer(norm):
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m.group(0))]
        nxt = _next_tokens(norm, m.end(), 1)
        if m.group("u") or (max(nums) <= 200 and not (nxt and nxt[0] in _COUNT_UNITS)):
            spans.append(m.span())
    return gsm, micron, spans


# --------------------------------------------------------------------------
# 4. Category classification: weighted phrase scoring
# --------------------------------------------------------------------------
_B = r"\b"
_RULES: list[tuple[str, float, str]] = [
    # ---- bopp_bag: BOPP-laminated woven bags
    ("bopp_bag", 12, r"\bbopp\s+(?:\w+\s+){0,3}?(?:bags?|sacks?|bori)\b"),
    ("bopp_bag", 5, r"\bbopp\b"),
    ("bopp_bag", 9, r"\b(?:lam|laminated|lamination|laminate)\s+(?:\w+\s+){0,2}?(?:woven|pp|hdpe|bags?|sacks?)\b"),
    ("bopp_bag", 5, r"\bbopp\b.*\bwoven\b|\bwoven\b.*\bbopp\b"),
    # ---- woven_sack: PP/HDPE woven sacks, cement bags, AD Star
    ("woven_sack", 12, r"\bad\s*star\b"),
    ("woven_sack", 8, r"\bcement\b"),
    ("woven_sack", 6, r"\bwoven\b"),
    ("woven_sack", 7, r"\bpp\s+(?:woven\s+)?(?:bags?|sacks?)\b"),
    ("woven_sack", 4, r"\b(?:hdpe|pp)\s+woven\b"),
    ("woven_sack", 5, r"\bsacks?\b"),
    ("woven_sack", 4, r"\b(?:raffia|gunny|bori)\b"),
    ("woven_sack", 5, r"\b(?:fertili[sz]er|urea|rice|sugar|wheat|grain|flour|atta|seeds?|feed|salt|pulses|dal|maize|cattle)\s+(?:\w+\s+)?(?:bags?|sacks?|bori)\b"),
    ("woven_sack", 4, r"\bhdpe\s+(?:bags?|sacks?)\b"),
    ("woven_sack", 1, r"\bbags?\b"),
    # ---- fibc: jumbo / bulk bags
    ("fibc", 14, r"\bfibc\b"),
    ("fibc", 9, r"\bjumbo\b"),
    ("fibc", 12, r"\bbulk\s+bags?\b"),
    ("fibc", 10, r"\bbig\s+bags?\b"),
    ("fibc", 10, r"\bsuper\s*sacks?\b"),
    ("fibc", 9, r"\bton(?:ne)?\s*bags?\b"),
    ("fibc", 5, r"\b(?:u\s*panel|baffle|cross\s*corner|swl|safe\s+working\s+load|bulker)\b"),
    ("fibc", 6, r"\bcontainer\s+liner"),
    ("fibc", -8, r"\bjumbo\s+rolls?\b"),
    # ---- corrugated_box
    ("corrugated_box", 10, r"\bcorrugat\w*"),
    ("corrugated_box", 7, r"\b[3579]\s*ply\b"),
    ("corrugated_box", 5, r"\bflute\b"),
    ("corrugated_box", 8, r"\bmaster\s+carton"),
    ("corrugated_box", 6, r"\b(?:shipper|shipping\s+(?:box|boxes|carton|cartons)|packing\s+(?:box|boxes)|brown\s+(?:box|boxes|carton|cartons))\b"),
    ("corrugated_box", 4.5, r"\bbox(?:es)?\b"),
    ("corrugated_box", 4, r"\bcartons?\b"),
    ("corrugated_box", 4, r"\bcarton\s+box"),
    ("corrugated_box", 6, r"\bcard\s*board\b"),
    # ---- mono_carton: printed folding cartons (board based)
    ("mono_carton", 16, r"\bmono\s*cartons?\b"),
    ("mono_carton", 12, r"\bfolding\s+(?:cartons?|box(?:es)?)\b"),
    ("mono_carton", 6, r"\bduplex\b"),
    ("mono_carton", 5, r"\b(?:ivory|fbb|sbs|art\s*card|grey\s*back|white\s*back|gable\s+top|tuck\s+end|crash\s+lock|lock\s+bottom|auto\s+bottom)\b"),
    ("mono_carton", 8, r"\b(?:pharma\w*|cosmetics?|medicines?|tablets?|perfume|toothpaste|cream|capsules?|syrup|ayurvedic|nutraceutical)\s+(?:\w+\s+)?(?:cartons?|box(?:es)?)\b"),
    ("mono_carton", 4, r"\b(?:offset|multicolou?r|uv|matt|matte|gloss|spot)\s+(?:\w+\s+)?cartons?\b"),
    ("mono_carton", 6, r"\brigid\s+box"),
    # ---- pouch
    ("pouch", 10, r"\bpouch\w*"),
    ("pouch", 6, r"\bstand\s*up\b"),
    ("pouch", 10, r"\bdoy\s*pack\b|\bdoypack\b"),
    ("pouch", 9, r"\bsachets?\b"),
    ("pouch", 6, r"\bretort\b"),
    ("pouch", 4, r"\bzip(?:per|\s*lock|lock)\b"),
    ("pouch", 6, r"\bspout\b"),
    ("pouch", 8, r"\b(?:3|three)\s+side\s+seal"),
    ("pouch", 5, r"\b(?:centre|center|back)\s+seal\b|\bpillow\b|\bflat\s+bottom\b|\bk\s*seal\b"),
    ("pouch", 5, r"\bmylar\b"),
    # ---- film
    ("film", 6, r"\bfilms?\b"),
    ("film", 9, r"\bshrink\w*"),
    ("film", 9, r"\bstretch\w*"),
    ("film", 4, r"\bwrap(?:s|ping)?\b"),
    ("film", 7, r"\bcling\b"),
    ("film", 6, r"\bpallet\s+wrap"),
    ("film", 8, r"\b(?:lldpe|ldpe|cpp|bopet|pet|metalli[sz]ed|bopp)\s+(?:film|rolls?|sheet|sheeting|tape)\b"),
    ("film", 2, r"\brolls?\b"),
    ("film", 4, r"\b(?:sheeting|bobbin|reel)\b"),
    ("film", 8, r"\bjumbo\s+rolls?\b"),
]
_RULES_C = [(c, w, re.compile(p)) for c, w, p in _RULES]


def _classify(norm: str, gsm: int | None):
    scores = {c: 0.0 for c in CATEGORIES}
    hits: list[tuple[str, float, str]] = []
    for cat, w, rx in _RULES_C:
        m = rx.search(norm)
        if m:
            scores[cat] += w
            hits.append((cat, w, m.group(0)))
    # board-weight cue: a "carton" quoted at >=200 gsm is folding-carton board, not corrugated
    if re.search(r"\bcartons?\b", norm) and gsm and gsm >= 200 and scores["corrugated_box"] <= 8:
        scores["mono_carton"] += 6
        hits.append(("mono_carton", 6, f"carton@{gsm}gsm"))
    best = max(scores.values())
    if best < 2:
        # nothing convincing: generic fallbacks by loose cues
        if re.search(r"\b(?:box|boxes|carton|cartons)\b", norm):
            return "corrugated_box", scores, hits
        if re.search(r"\b(?:roll|rolls|wrap|sheet|sheets)\b", norm):
            return "film", scores, hits
        return _DEFAULT_CATEGORY, scores, hits
    order = list(CATEGORIES)
    top = [c for c in order if scores[c] == best]
    return top[0], scores, hits


# --------------------------------------------------------------------------
# 5. Quantity & unit
# --------------------------------------------------------------------------
_WEIGHT_UNITS = {"kg": ("kg", 1.0), "ton": ("mt", 1000.0), "mt": ("mt", 1000.0),
                 "quintal": ("kg", 100.0)}
_CONTAINERS = {"bag", "bags", "sack", "sacks", "pouch", "packing", "pack", "packet",
               "packets", "pkt", "bori", "box", "boxes", "carton", "cartons", "liner"}
_CAP_AFTER = {"capacity", "cap", "capcity", "capasity", "swl", "load", "loading", "holding",
              "wala", "wale", "wali", "walla", "ka", "ki", "ke", "each", "per", "weight",
              "wt", "size", "bearing", "fill", "filling", "x"}
_CAP_BEFORE = {"capacity", "cap", "capcity", "capasity", "swl", "load", "loading", "holding",
               "weighing", "upto", "size", "each", "fill", "filling", "weight", "wt"}
_BAG_MODS = {"pp", "hdpe", "woven", "laminated", "laminate", "lam", "printed", "plastic",
             "jumbo", "big", "bulk", "cement", "bopp", "ad", "star", "new", "empty", "plain",
             "fibc", "super", "white", "blue", "paper", "kraft", "poly", "non", "valve",
             "open", "mouth", "sugar", "rice", "wheat", "fertilizer", "grain", "flour", "atta",
             "urea", "salt", "feed", "small", "standard", "normal", "stand", "up", "zipper"}
_QTY_CUES = {"qty", "quantity", "need", "needed", "required", "require", "requirement",
             "chahiye", "order", "moq", "total", "want", "wanted", "buy", "purchase",
             "req", "atleast", "min", "minimum", "about", "around", "approx", "lagega", "of"}
_CAT_NOUNS = _COUNT_UNITS | {"film", "wrap", "mono", "jumbo", "fibc", "bopp", "woven", "corrugated"}


def _weight_is_capacity(kg: float, prev: list[str], nxt: list[str], category: str) -> bool:
    """Is 'N kg' describing what a single bag/box holds rather than what is ordered?"""
    if nxt and nxt[0] in _CAP_AFTER:
        return True
    if prev and prev[-1] in _CAP_BEFORE:
        return True
    if len(prev) >= 2 and prev[-2] in _CAP_BEFORE:  # "capacity of 1 ton"
        return True
    j = 0
    while j < len(nxt) and j < 4 and nxt[j] in _BAG_MODS:
        j += 1
    container_after = j < len(nxt) and nxt[j] in _CONTAINERS
    container_before = bool(prev) and (
        prev[-1] in _CONTAINERS or (len(prev) >= 2 and prev[-1] in {"of", "x"} and prev[-2] in _CONTAINERS))
    if container_after or container_before:
        if category == "fibc":
            return 250 <= kg <= 3000
        return kg <= 100
    return False


def _extract_quantity(norm: str, masked_spans, category: str):
    cands: list[dict] = []
    for m in _NUM_RE.finditer(norm):
        if any(s <= m.start() < e for s, e in masked_spans):
            continue
        n1 = float(m.group("n1"))
        m1 = m.group("m1") or (m.group("m2") if m.group("n2") else None)
        value = n1 * _MULT.get(m1, 1)
        if value <= 0:
            continue
        prev = _prev_tokens(norm, m.start())
        nxt = _next_tokens(norm, m.end())
        # pincode right after a place name / 6-digit plain number starting 1-8 next to a city
        if (re.fullmatch(r"[1-8]\d{5}", m.group("n1")) and not m1 and not (nxt and (nxt[0] in _COUNT_UNITS or nxt[0] in _WEIGHT_UNITS))
                and prev and (prev[-1] in _PLACE_WORDS or prev[-1] == "pin")):
            continue
        if not m1 and 1990 <= value <= 2100 and prev and prev[-1] in {"in", "by", "year", "since", "for", "dated"}:
            continue
        first = nxt[0] if nxt else ""
        cue = any(t in _QTY_CUES for t in prev[-2:])
        unit = "pcs"
        if first in _WEIGHT_UNITS:
            u, factor = _WEIGHT_UNITS[first]
            kg = value * (1000.0 if u == "mt" else factor if first == "quintal" else 1.0)
            if _weight_is_capacity(kg, prev, nxt[1:], category):
                continue  # fill weight / bag capacity: never the order quantity
            if first == "quintal":
                value, unit = value * 100, "kg"
            else:
                unit = u
            score = 6 + (2 if cue else 0) + (1 if m1 else 0) + (1 if category == "film" else 0)
        elif first in _COUNT_UNITS:
            score = 10 + (2 if cue else 0)
        else:
            window = []
            for t in nxt:
                if t == "," or re.match(r"\d", t):
                    break
                window.append(t)
            if any(t in _CAT_NOUNS for t in window[:4]):
                score = 7 + (2 if cue else 0) + (1 if m1 else 0)
            else:
                score = 3 + (2 if cue else 0) + (1 if m1 else 0)
        cands.append({"value": value, "unit": unit, "score": score, "pos": m.start(),
                      "kg": value * (1000.0 if unit == "mt" else 1.0)})
    if not cands:
        return None, "pcs", cands
    if category != "film" and len(cands) > 1:
        # "bopp 50 kg 10000": a small unit weight next to another figure is bag capacity
        for c in cands:
            if c["unit"] == "kg" and c["kg"] <= (3000 if category == "fibc" else 100):
                c["score"] -= 5
    best = max(cands, key=lambda c: (c["score"], -c["pos"]))
    v = best["value"]
    v = int(v) if float(v).is_integer() else round(v, 4)
    return v, best["unit"], cands


# --------------------------------------------------------------------------
# 6. Location
# --------------------------------------------------------------------------
_CITIES = """
mumbai,bombay,delhi,new delhi,bangalore,bengaluru,hyderabad,secunderabad,chennai,madras,
kolkata,calcutta,pune,ahmedabad,surat,jaipur,lucknow,kanpur,nagpur,indore,thane,bhopal,
visakhapatnam,vizag,patna,vadodara,baroda,ghaziabad,ludhiana,agra,nashik,nasik,faridabad,
meerut,rajkot,kalyan,vasai,virar,varanasi,srinagar,aurangabad,dhanbad,amritsar,navi mumbai,
allahabad,prayagraj,ranchi,howrah,coimbatore,jabalpur,gwalior,vijayawada,jodhpur,madurai,
raipur,kota,guwahati,chandigarh,solapur,hubli,hubballi,dharwad,bareilly,moradabad,mysore,
mysuru,gurgaon,gurugram,aligarh,jalandhar,tiruchirappalli,trichy,bhubaneswar,bhubaneshwar,
salem,warangal,guntur,bhiwandi,saharanpur,gorakhpur,bikaner,amravati,noida,greater noida,
jamshedpur,bhilai,cuttack,firozabad,kochi,cochin,ernakulam,bhavnagar,dehradun,durgapur,
asansol,nanded,kolhapur,ajmer,akola,gulbarga,kalaburagi,jamnagar,ujjain,siliguri,jhansi,
ulhasnagar,nellore,jammu,sangli,belgaum,belagavi,mangalore,mangaluru,tirunelveli,malegaon,
gaya,jalgaon,udaipur,davanagere,kozhikode,calicut,kurnool,rajahmundry,bokaro,bellary,
ballari,patiala,agartala,bhagalpur,muzaffarnagar,latur,dhule,tirupati,rohtak,korba,bhilwara,
berhampur,muzaffarpur,ahmednagar,mathura,kollam,kadapa,anantapur,bilaspur,sambalpur,
shahjahanpur,satara,bijapur,vijayapura,rampur,shimoga,shivamogga,chandrapur,junagadh,
thrissur,alwar,bardhaman,burdwan,kakinada,nizamabad,parbhani,tumkur,tumakuru,khammam,
panipat,darbhanga,aizawl,dewas,karnal,bathinda,jalna,eluru,purnia,satna,sonipat,sonepat,
farrukhabad,sagar,rourkela,durg,imphal,ratlam,hapur,arrah,karimnagar,etawah,bharatpur,
begusarai,gandhidham,tiruppur,tirupur,pondicherry,sikar,thoothukudi,tuticorin,rewa,
mirzapur,raichur,ramagundam,silchar,haridwar,tenali,nagercoil,sri ganganagar,thanjavur,
bulandshahr,katni,sambhal,singrauli,nadiad,yamunanagar,bidar,munger,panchkula,burhanpur,
kharagpur,dindigul,gandhinagar,hospet,malda,ongole,deoghar,chapra,haldia,khandwa,nandyal,
morena,amroha,anand,bhind,bhiwani,ambala,morbi,mehsana,palanpur,himmatnagar,bharuch,
ankleshwar,vapi,valsad,navsari,silvassa,daman,diu,bhuj,mundra,kandla,porbandar,
surendranagar,amreli,godhra,dahod,gondal,wankaner,halol,kalol,sanand,dholera,hosur,
krishnagiri,erode,karur,namakkal,vellore,ranipet,ambur,kanchipuram,chengalpattu,
sriperumbudur,tiruvallur,tiruvannamalai,cuddalore,villupuram,pudukkottai,sivakasi,
virudhunagar,rajapalayam,theni,ooty,coonoor,palakkad,kannur,kasaragod,alappuzha,kottayam,
thiruvananthapuram,trivandrum,malappuram,hassan,mandya,chitradurga,udupi,bagalkot,koppal,
gadag,haveri,bhadravati,hoskote,nelamangala,peenya,doddaballapur,hindupur,chittoor,
machilipatnam,bhimavaram,srikakulam,vizianagaram,adoni,siddipet,mahbubnagar,nalgonda,
sangareddy,patancheru,jeedimetla,kukatpally,okhla,narela,bawana,mundka,bhiwadi,manesar,
sohna,rewari,jhajjar,bahadurgarh,palwal,kundli,kurukshetra,hisar,fatehabad,sirsa,jind,
kaithal,mohali,zirakpur,derabassi,baddi,parwanoo,solan,shimla,kangra,pathankot,gurdaspur,
hoshiarpur,kapurthala,moga,firozpur,muktsar,sangrur,barnala,mandi gobindgarh,khanna,
rajpura,roorkee,rudrapur,kashipur,haldwani,kotdwar,modinagar,sahibabad,khurja,hathras,
etah,kasganj,pilibhit,lakhimpur,sitapur,hardoi,unnao,fatehpur,raebareli,sultanpur,
faizabad,ayodhya,barabanki,gonda,basti,deoria,azamgarh,jaunpur,ghazipur,ballia,bhadohi,
banda,hamirpur,orai,mandsaur,neemuch,khargone,pithampur,mandideep,vidisha,hoshangabad,
itarsi,betul,chhindwara,seoni,balaghat,damoh,chhatarpur,datia,shivpuri,guna,raisen,sehore,
dhar,jhabua,barwani,raigarh,jagdalpur,ambikapur,rajnandgaon,hazaribagh,giridih,katihar,
sasaram,hajipur,jalpaiguri,cooch behar,puri,balasore,jharsuguda,angul,dibrugarh,jorhat,
tezpur,nagaon,tinsukia,shillong,kohima,dimapur,itanagar,gangtok,wardha,yavatmal,gondia,
bhandara,ratnagiri,chiplun,panvel,taloja,tarapur,boisar,palghar,dombivli,ambernath,
badlapur,chakan,ranjangaon,talegaon,pimpri,chinchwad,pimpri chinchwad,hinjewadi,baramati,
sinnar,manmad,shirdi,beed,osmanabad,hingoli,washim,buldhana,nandurbar,sindhudurg,mahad,
khopoli,alibag,churu,jhunjhunu,nagaur,barmer,jaisalmer,sirohi,abu road,chittorgarh,
neemrana,behror,kishangarh,beawar,tonk,dausa,banswara,dungarpur,bundi,jhalawar,baran,
hanumangarh,port blair,leh,kargil,anantnag,baramulla,kathua,udhampur,
abu dhabi,dubai,sharjah,ajman,doha,riyadh,jeddah,dammam,muscat,colombo,dhaka,chittagong,
kathmandu,nairobi,mombasa,lagos,accra,dar es salaam,addis ababa,cairo,bangkok,jakarta,
kuala lumpur,karachi,lahore,hanoi,london,new york,toronto,kuwait city,manama,tehran,
istanbul,johannesburg,durban,cape town,casablanca,tashkent,yangon,phnom penh,manila
"""
_STATES = """
andhra pradesh,arunachal pradesh,assam,bihar,chhattisgarh,goa,gujarat,haryana,
himachal pradesh,himachal,jharkhand,karnataka,kerala,madhya pradesh,maharashtra,manipur,
meghalaya,mizoram,nagaland,odisha,orissa,punjab,rajasthan,sikkim,tamil nadu,tamilnadu,
telangana,tripura,uttar pradesh,uttarakhand,uttaranchal,west bengal,jammu and kashmir,
jammu & kashmir,kashmir,ladakh,andaman,lakshadweep,dadra and nagar haveli,kutch,
saurashtra,vidarbha,marathwada,konkan,bengal
"""
_COUNTRIES = """
india,usa,uae,uk,america,united states,united kingdom,united arab emirates,saudi arabia,
saudi,oman,qatar,kuwait,bahrain,iran,iraq,israel,jordan,lebanon,yemen,turkey,egypt,libya,
nigeria,ghana,kenya,tanzania,uganda,ethiopia,sudan,south africa,zambia,zimbabwe,mozambique,
angola,senegal,ivory coast,cameroon,morocco,algeria,tunisia,bangladesh,nepal,bhutan,
sri lanka,srilanka,pakistan,afghanistan,myanmar,thailand,vietnam,cambodia,laos,malaysia,
singapore,indonesia,philippines,china,japan,korea,australia,new zealand,germany,france,
italy,spain,portugal,netherlands,belgium,sweden,norway,denmark,finland,poland,russia,
ukraine,canada,mexico,brazil,argentina,chile,peru,colombia,uzbekistan,kazakhstan,maldives,
mauritius,hong kong,taiwan
"""


def _split(s: str) -> list[str]:
    return [x.strip() for x in s.replace("\n", "").split(",") if x.strip()]


_GAZ: dict[str, int] = {}  # name -> rank (3 city, 2 state, 1 country)
for _n in _split(_COUNTRIES):
    _GAZ[_n] = 1
for _n in _split(_STATES):
    _GAZ[_n] = 2
for _n in _split(_CITIES):
    _GAZ[_n] = 3
_PLACE_WORDS = {w for name in _GAZ for w in name.split()}
_GAZ_BY_LETTER: dict[str, list[str]] = {}
for _n in _GAZ:
    _GAZ_BY_LETTER.setdefault(_n[0], []).append(_n)
_ACRONYMS = {"uae", "usa", "uk", "ncr"}
_STATE_ABBR = {"UP", "MP", "AP", "TN", "HP", "WB", "HR", "RJ", "MH", "GJ", "KA", "OD", "CG", "JK"}

# Words that are never (the start of) a location: product, spec, unit, filler, Hinglish, etc.
_NON_LOC = set("""
a an the and or of for to from in at on by with without per as is are was be been am do does did
i we you he she it they me my our your us this that these those there here also only just very
need needs needed require required requires requirement requirements want wanted wants looking
look looks buy buying purchase order orders ordering supply supplier suppliers manufacturer
manufacturers quote quotation quotations rate rates price prices pricing cost best good quality
grade virgin recycled sample samples please pls plz kindly thanks thank regards hi hello hey dear
sir madam madame mr mrs ms dr team urgent urgently asap immediately soon today tomorrow monthly
weekly daily regular regularly bulk wholesale retail export import domestic local new old fresh
send share give get provide kindly confirm reply call contact whatsapp email mail number phone
mobile details detail info information price list catalog catalogue moq lead time delivery
deliver delivered dispatch dispatched shipping ship shipped shipment transport freight payment
advance credit cash gst tax invoice bill ex fob cif works factory plant godown warehouse site
location loc city town place destination address state country pin pincode area
chahiye chahiyen chahie chaiye cahiye chahta chahte chahti hai hain ho hoga hogi honge tha thi
the hum humko hume mujhe mereko mujhko humein aap aapko aapka aapke tum tumhe apna apni apne
hamara hamari humara ka ki ke ko se mein me mai men par pe tak aur ya bhi sirf abhi jaldi turant
kitna kitne kitni kya kab kahan kaun koi kuch sab sabhi saath liye liya lena dena do dijiye
dijiy kijiye karo kar karna kare bhejo bhej bhejiye bhejna batao bataiye bataye bata rate bhav
bhaav dam daam kimat wala wale wali waala ji bhai bhaiya saheb sahab sahib kripya
lagega lagenge lagta lagti zarurat zaroorat jarurat jaroorat chahiyye
bag bags sack sacks bori box boxes carton cartons pouch pouches sachet sachets film films roll
rolls wrap wrapping wraps shrink stretch cling pallet jumbo fibc bulk liner liners packaging
packing pack packs packet packets pkt pkts material materials product products item items
bopp pp hdpe ldpe lldpe cpp pet pe pvc bop woven non nonwoven laminated lamination laminate lam
plain printed print printing colour color colours colors white blue red green black yellow natural
transparent clear opaque matt matte glossy gloss kraft brown paper board duplex ivory uv mono
corrugated ply flute cardboard cement ad star adstar fertilizer fertiliser urea sugar rice wheat
grain grains flour atta seed seeds feed salt pulses dal maize cattle chemical chemicals pharma
pharmaceutical cosmetic cosmetics food snacks spices masala tea coffee garment garments textile
panel panels baffle type types swl u gusset gusseted handle handles loop loops seal sealed side sleeve sleeves layer layers colour hand machine jumbo
zipper zip lock ziplock stand up spout retort mylar standup doypack doy
pcs pc piece pieces nos no unit units kg kgs mt mts ton tons tonne tonnes quintal kilo gram grams
gm gms gsm micron microns mic mics mm cm inch inches ft feet meter meters metre mtr mtrs ml ltr
litre litres lakh lakhs lac lacs hazaar hazar thousand hundred crore million sau k cr
width length height thickness thick thin weight capacity cap size sizes dimension dimensions
qty quantity total approx approximately around about atleast least minimum maximum min max
upto up load loading swl each every any all some more less same other another next last first
one two three four five six seven eight nine ten dozen lot lots set sets pair pairs
rs inr usd rupees rupee dollar dollars budget target
monthly yearly annual contract trial repeat sample
india indian made make china chinese
name company co ltd pvt private limited llp inc corp enterprises enterprise traders trader
industries industry exports export packers packaging solutions solution agency agencies
supplies suppliers dealer dealers distributor distributors marketing sales
if but so then than too can could will would should may might must shall have has had not yes
okay ok ya yes no nope sure thanks ty tq
""".split())
_NON_LOC -= set(_GAZ)  # a real place name always wins over a stop-word
_KNOWN = _NON_LOC | _PLACE_WORDS | set(_FUZZY_TARGETS) | set(_TYPOS.values()) | set(_WORD_VALUES) | _WORD_MULTS

_STRONG_PRE = {"deliver", "delivery", "delivered", "dispatch", "ship", "shipping", "send",
               "supply", "destination", "location", "loc", "city", "town", "site", "place",
               "deliverable", "delivering", "reach", "reaching", "freight"}
_SOFT_PRE = {"to", "at", "in", "near", "around", "within"}
_POST_CUES = {"mein", "me", "mai", "men", "tak", "ke", "ko", "delivery", "wale", "wala", "pe",
              "ka", "ki", "se"}
_FROM_PRE = {"from", "ex"}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")


def _canon(name: str) -> str:
    return name.upper() if name in _ACRONYMS else " ".join(w.capitalize() for w in name.split())


@lru_cache(maxsize=4096)
def _fuzzy_place(phrase: str):
    """Typo-tolerant gazetteer lookup ('banglore', 'ahmadabad', 'navi mumbi')."""
    if len(phrase) < 5 or phrase in _NON_LOC:
        return None
    cands = [n for n in _GAZ_BY_LETTER.get(phrase[0], ()) if abs(len(n) - len(phrase)) <= 2]
    best = get_close_matches(phrase, cands, n=1, cutoff=0.87)
    if not best:
        return None
    # guard against matching a different but similar real word
    return best[0] if SequenceMatcher(None, phrase, best[0]).ratio() >= 0.87 else None


def _extract_location(text: str) -> str | None:
    light = unicodedata.normalize("NFKC", text)
    toks = [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(light)]
    if not toks:
        return None
    low = [t[0].lower().strip("'") for t in toks]
    n = len(toks)

    def joined(i: int, j: int) -> bool:  # tokens i..j separated only by spaces/hyphens
        return all(re.fullmatch(r"[\s\-]+", light[toks[k][2]:toks[k + 1][1]]) for k in range(i, j))

    def between(i: int) -> str:  # raw text between token i and i+1
        return light[toks[i][2]:toks[i + 1][1]] if i + 1 < n else ""

    case_signal = text != text.lower() and text != text.upper()  # all-lower / ALL-CAPS carry no case info
    def cue_score(i: int, j: int) -> float:
        s = 0.0
        pre = low[max(0, i - 3):i]
        if pre:
            if pre[-1] in _STRONG_PRE or (len(pre) >= 2 and pre[-2] in _STRONG_PRE and pre[-1] in _SOFT_PRE):
                s += 5
            elif pre[-1] in _SOFT_PRE:
                s += 3
            elif pre[-1] in _FROM_PRE:
                s -= 5
        if j + 1 < n and low[j + 1] in _POST_CUES and "," not in between(j):
            s += 2 if low[j + 1] != "ka" else 0
        return s

    mentions: list[dict] = []
    used = [False] * n
    for size in (3, 2, 1):  # longest phrase first
        for i in range(n - size + 1):
            if any(used[i:i + size]) or not joined(i, i + size - 1):
                continue
            phrase = " ".join(low[i:i + size])
            canon = phrase if phrase in _GAZ else None
            fuzzy = False
            if canon is None and not any(w in _NON_LOC or w in _KNOWN and w not in _PLACE_WORDS for w in low[i:i + size]):
                # typo match is only trusted with a second signal: capitalised in mixed-case
                # text, or a grammar cue ("to X", "X mein", "deliver X")
                cand = _fuzzy_place(phrase)
                if cand and (
                        (case_signal and toks[i][0][0].isupper())
                        or cue_score(i, i + size - 1) > 0):
                    canon, fuzzy = cand, True
            if canon:
                for k in range(i, i + size):
                    used[k] = True
                shown = _canon(canon) if fuzzy or canon in _ACRONYMS else _canon(phrase)
                mentions.append({"i": i, "j": i + size - 1, "name": shown, "rank": _GAZ[canon], "kind": "gaz"})

    # Bare state abbreviations are only trusted when written in capitals inside mixed-case text
    if case_signal:
        for i in range(n):
            if not used[i] and toks[i][0] in _STATE_ABBR:
                mentions.append({"i": i, "j": i, "name": toks[i][0], "rank": 2, "kind": "abbr"})
                used[i] = True

    # Made in India / make in india is not a delivery location
    mentions = [m for m in mentions
                if not (m["name"] == "India" and m["i"] >= 2 and low[m["i"] - 1] == "in" and low[m["i"] - 2] in {"made", "make"})]

    best_m = None
    if mentions:
        for m in mentions:
            m["score"] = m["rank"] + cue_score(m["i"], m["j"]) - 0.01 * m["i"]
        best_m = max(mentions, key=lambda m: m["score"])
        if best_m["rank"] >= 3:  # a known city wins outright
            return best_m["name"]

    # --- unseen place: rely on grammar cues rather than a word list
    all_lower = not case_signal
    cands = []
    for i in range(n):
        w = low[i]
        if used[i] or len(w) < 3 or w in _NON_LOC or w in _KNOWN or w in _TYPOS:
            continue
        orig_cap = toks[i][0][0].isupper() and case_signal
        j = i
        while (j + 1 < n and j - i < 2 and joined(j, j + 1) and low[j + 1] not in _NON_LOC
               and low[j + 1] not in _KNOWN and toks[j + 1][0][0].isupper() == orig_cap and len(low[j + 1]) >= 3):
            j += 1
        pre = low[max(0, i - 3):i]
        score = 0.0
        if pre and (pre[-1] in _STRONG_PRE or (len(pre) >= 2 and pre[-2] in _STRONG_PRE and pre[-1] in _SOFT_PRE)):
            score = 6
        elif pre and pre[-1] in _SOFT_PRE and (orig_cap or (pre[-1] == "to" and all_lower)):
            score = 4 if orig_cap else 3
        elif pre and pre[-1] in _FROM_PRE:
            continue
        if j + 1 < n and low[j + 1] in _POST_CUES - {"ka", "ki", "se", "ke", "ko"} and "," not in between(j):
            score = max(score, 4)
        # a standalone capitalised fragment at the end / between commas: "..., 65 gsm, Hosur"
        if score == 0 and orig_cap and not all_lower:
            after = light[toks[j][2]:]
            before = light[:toks[i][1]]
            seg_end = re.match(r"\s*(?:$|[,;\n.])", after) is not None
            seg_start = re.search(r"(?:^|[,;\n])\s*$", before) is not None and i > 0
            sig = low[i - 1] if i else ""
            if seg_end and (seg_start or j == n - 1) and sig not in {"regards", "thanks", "thank", "sir", "madam", "mr", "mrs", "ms", "dr", "name"} \
                    and not re.search(r"\b(regards|thanks|thank you)\b", before.lower()):
                score = 2
        if score:
            cands.append((score - 0.01 * i, " ".join(w.capitalize() for w in low[i:j + 1])))
    if cands:
        score, name = max(cands)
        # "Raigad, Maharashtra": an unseen town with a grammar cue is more specific than a state
        if best_m is None or score >= 3.5:
            return name
    return best_m["name"] if best_m else None


# --------------------------------------------------------------------------
# 7. Public API
# --------------------------------------------------------------------------
def _analyse(text: str) -> dict:
    """Run the full pipeline and keep the intermediate evidence (used by the CLI trace)."""
    norm = _words_to_digits(_fix_typos(_normalize(text)))
    # first pass without category so spec extraction can mask numbers, second pass for 'gms'
    gsm0, _mic0, _sp0 = _extract_specs(norm, "")
    category, scores, hits = _classify(norm, gsm0)
    gsm, micron, spans = _extract_specs(norm, category)
    qty, unit, cands = _extract_quantity(norm, spans, category)
    return {"norm": norm, "category": category, "scores": scores, "hits": hits,
            "gsm": gsm, "micron": micron, "quantity": qty, "unit": unit, "cands": cands}


def parse_rfq(text: str) -> dict:
    """Turn a messy buyer message into the six structured RFQ fields."""
    out = {"category": _DEFAULT_CATEGORY, "quantity": None, "unit": "pcs",
           "gsm": None, "micron": None, "location": None}
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    try:
        a = _analyse(text)
        out.update(category=a["category"], quantity=a["quantity"], unit=a["unit"],
                   gsm=a["gsm"], micron=a["micron"])
    except Exception:  # never crash the caller on odd input
        pass
    try:
        out["location"] = _extract_location(text)
    except Exception:
        pass
    return out


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) or "bopp bags chahiye 40 hazaar, 65 gsm, Nagpur"
    info = _analyse(msg)
    print(parse_rfq(msg))
    print("normalised :", info["norm"])
    print("category   :", {k: v for k, v in info["scores"].items() if v})
    print("candidates :", info["cands"])
