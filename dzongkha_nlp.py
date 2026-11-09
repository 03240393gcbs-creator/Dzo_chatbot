"""Dzongkha text tools: normalisation, spelling checks, simple sentence-structure checks.

Everything here is rule/lexicon based (no model download) so it is fast, testable
and its behaviour can be explained in an NLP report.

Pipeline used by the app on every Dzongkha reply:
    normalize()  ->  apply_variant_fixes()  ->  check_syllables()/check_structure()
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

TSHEG = "\u0F0B"
SHAD = "\u0F0D"
TIB_CHAR = r"\u0F00-\u0FFF"
TIB = re.compile(f"[{TIB_CHAR}]")

# ASCII digit -> Dzongkha digit (the Zenodo 6271560 dataset covers these 10 classes)
DIGITS = str.maketrans("0123456789", "༠༡༢༣༤༥༦༧༨༩")
DZ_DIGIT_NAMES = {
    "0": "ཀླད་ཀོར", "1": "གཅིག", "2": "གཉིས", "3": "གསུམ", "4": "བཞི",
    "5": "ལྔ", "6": "དྲུག", "7": "བདུན", "8": "བརྒྱད", "9": "དགུ", "10": "བཅུ",
}

# ------------------------------------------------------------------ normalise
def normalize(text: str, digits: bool = True, strip_foreign: bool = True) -> str:
    """Return cleanly formatted Dzongkha.

    - Unicode NFC, tsheg variants unified
    - no spaces between syllables (Dzongkha is written with tsheg only)
    - exactly one space after a shad when text continues
    - single shad (། not །།) at sentence end, and a shad is always present
    - optional: ASCII digits -> ༠-༩, remove stray Latin / Chinese characters
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = t.replace("\u0F0C", TSHEG).replace("\u00A0", " ")
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)
    if strip_foreign:
        t = re.sub(r"[\u4e00-\u9fff]", "", t)                 # Chinese
        t = re.sub(r"(?<=[" + TIB_CHAR + r"])\s*[A-Za-z]+\s*(?=[" + TIB_CHAR + r"])", " ", t)
    if digits:
        t = t.translate(DIGITS)
    # collapse repeated shads and the "་།" / " །" patterns
    t = re.sub(f"{TSHEG}?\\s*{SHAD}(\\s*{SHAD})+", SHAD, t)
    t = re.sub(f"\\s+{SHAD}", SHAD, t)
    # remove spaces that sit between Tibetan syllables (after a tsheg, or between letters)
    t = re.sub(f"{TSHEG}\\s+(?=[{TIB_CHAR}])", TSHEG, t)
    # normalise whitespace after shad
    t = re.sub(f"{SHAD}\\s*(?=[{TIB_CHAR}])", SHAD + " ", t)
    t = re.sub(r"[ \t]+", " ", t).strip()
    # ensure a final shad if the text ends in a Tibetan letter
    if t and TIB.match(t[-1]) and t[-1] not in (SHAD,):
        if t[-1] == TSHEG:
            t = t[:-1]
        t += SHAD
    return t


# --------------------------------------------------------- known wrong variants
# Classical-Tibetan / wrong spellings that machine translation and older data produce
# -> the Dzongkha spelling.  Whole-syllable replacement only (never inside a longer syllable).
DEFAULT_VARIANTS = {
    "ཡིན": "ཨིན",       # is / am / are  (Dzongkha copula)
    "མིན": "མེན",       # is not
    "ཐུགས་རྗེ་ཆེ": "བཀའ་དྲིན་ཆེ",   # thank you (Tibetan form -> Dzongkha)
    "ཀུ་ཟུ་ཟང་ཕོ་ལ": "ཀུ་ཟུ་ཟང་པོ་ལ",
    "ཉིས": "གཉིས",      # two (missing prefix)
    "སུམ": "གསུམ",      # three (missing prefix)
}


def split_syllables(text: str) -> list[str]:
    return [s for s in re.split(f"[{TSHEG}{SHAD}\\s]+", text) if s and TIB.search(s)]


def apply_variant_fixes(text: str, variants: dict[str, str] | None = None) -> tuple[str, list[tuple[str, str]]]:
    """Replace known wrong spellings.  Returns (fixed_text, [(old, new), ...])."""
    variants = variants or DEFAULT_VARIANTS
    changes: list[tuple[str, str]] = []
    # longest keys first so multi-syllable phrases win
    for old in sorted(variants, key=len, reverse=True):
        new = variants[old]
        # boundary = start, tsheg, shad or whitespace before; tsheg, shad, whitespace or end after
        pat = re.compile(f"(?:(?<=^)|(?<=[{TSHEG}{SHAD}\\s])){re.escape(old)}(?=$|[{TSHEG}{SHAD}\\s])")
        if pat.search(text):
            text = pat.sub(new, text)
            changes.append((old, new))
    return text, changes


# ------------------------------------------------------------ syllable validity
BASE = r"[\u0F40-\u0F6C]"
SUBJ = r"[\u0F90-\u0FBC]"
VOWEL = r"[\u0F71-\u0F7D\u0F80-\u0F84]"
# one syllable: optional stack(s) of base + subjoined, vowels, marks
_SYL = re.compile(
    rf"^(?:{BASE}{SUBJ}*[\u0F35\u0F37\u0F39\u0F3E\u0F3F]*(?:{VOWEL}){{0,2}}[\u0F7E\u0F7F\u0F83\u0F86\u0F87]*)+$"
)
_DIGIT_SYL = re.compile(r"^[\u0F20-\u0F33]+$")
_MULTI_TOP_VOWEL = re.compile(r"[\u0F72\u0F74\u0F7A\u0F7C][\u0F72\u0F74\u0F7A\u0F7C]")


def syllable_is_well_formed(syl: str) -> bool:
    """Structural check only (vowel signs on a consonant, no doubled vowels, ...)."""
    if _DIGIT_SYL.match(syl):
        return True
    if _MULTI_TOP_VOWEL.search(syl):
        return False
    return bool(_SYL.match(syl))


# --------------------------------------------------------------------- lexicon
class Lexicon:
    """Syllable lexicon built from verified text (answer bank, cleaned dataset,
    and any extra corpus file you add, e.g. text from the CST Dzongkha resources)."""

    def __init__(self):
        self.counts: dict[str, int] = {}

    def add_text(self, text: str):
        for s in split_syllables(text):
            self.counts[s] = self.counts.get(s, 0) + 1

    def add_file(self, path: str | Path):
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                self.add_text(line)

    def __contains__(self, syl: str) -> bool:
        return syl in self.counts

    def __len__(self):
        return len(self.counts)

    def suggest(self, syl: str, n: int = 3, cutoff: float = 0.7) -> list[str]:
        return difflib.get_close_matches(syl, self.counts.keys(), n=n, cutoff=cutoff)


# ------------------------------------------------------------ structure heuristics
# Dzongkha is SOV: a finished declarative/interrogative sentence ends in a verb, copula
# or auxiliary.  These are the common sentence-final forms.
FINAL_FORMS = {
    "ཨིན", "མེན", "ཡོད", "མེད", "འདུག", "མིན", "སྨོ", "ལགས", "ཨིན་པས", "ཨིན་ན",
    "ཨིན་ནམ", "ཨིན་མས", "དགོ", "ཚུགས", "འོང", "འགྱོ", "སོ", "རུང", "ཨིན་ན", "གི", "གེ",
    "པས", "པའི་ཨིན",
}
_FINAL_LAST_SYL = {f.split(TSHEG)[-1] for f in FINAL_FORMS} | {
    "ཨིན", "མེན", "ཡོད", "མེད", "འདུག", "སྨོ", "ལགས", "དགོ", "ཚུགས", "འོང", "འགྱོ",
    "ཨིན", "ན", "ནམ", "མས", "པས", "ག", "ཅི", "ཏོ", "རུ", "གོ", "ནུག", "སོ", "པ", "ཁམས",
    "ལ", "ཐོན", "སོན", "བྱུང", "བཏང", "སྒྲིག", "ཞུ", "ཕེབས", "གནང", "སྦེ", "ནི", "གེ", "ལོ",
}

# location noun directly followed by an existence verb needs the locative ལུ
_LOC_FIX = re.compile(f"(ནང|སྟེང|འོག|མདུན|རྒྱབ|ཁར)({TSHEG})(?=(ཡོད|མེད|འདུག))")


@dataclass
class Report:
    original: str
    fixed: str
    changes: list[tuple[str, str]] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)
    unknown: list[tuple[str, list[str]]] = field(default_factory=list)
    structure: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.malformed or self.structure)


def check_structure(text: str) -> list[str]:
    """Heuristic warnings about sentence structure (SOV, verb-final, locative)."""
    warns = []
    sents = [s.strip() for s in re.split(SHAD, text) if s.strip()]
    for s in sents:
        syls = split_syllables(s)
        if not syls:
            continue
        if len(syls) >= 5 and syls[-1] not in _FINAL_LAST_SYL:     # shorter strings are phrases/greetings, not full sentences
            warns.append(f"Sentence does not end in a verb/copula (Dzongkha is verb-final): …{TSHEG.join(syls[-3:])}")
        if _LOC_FIX.search(s + TSHEG):
            warns.append("Location word followed directly by a verb; Dzongkha normally needs ལུ་ (e.g. ནང་ལུ་ཡོད)")
    return warns


def autofix_structure(text: str) -> str:
    """Safe, rule-based structure fix: add the locative ལུ between ནང་/སྟེང་... and ཡོད/མེད/འདུག."""
    return _LOC_FIX.sub(lambda m: f"{m.group(1)}{TSHEG}ལུ{TSHEG}", text)


def analyse(text: str, lexicon: Lexicon | None = None, auto_fix: bool = True) -> Report:
    """Full clean-up + diagnostics for one Dzongkha string."""
    t = normalize(text)
    changes: list[tuple[str, str]] = []
    if auto_fix:
        t, changes = apply_variant_fixes(t)
        t2 = autofix_structure(t)
        if t2 != t:
            changes.append(("ནང་ཡོད", "ནང་ལུ་ཡོད"))
            t = t2
        t = normalize(t)
    malformed = [s for s in split_syllables(t) if not syllable_is_well_formed(s)]
    unknown = []
    if lexicon is not None and len(lexicon) > 50:       # only meaningful with a decent lexicon
        for s in dict.fromkeys(split_syllables(t)):
            if s not in lexicon and not _DIGIT_SYL.match(s):
                unknown.append((s, lexicon.suggest(s)))
    return Report(text, t, changes, malformed, unknown, check_structure(t))


# ------------------------------------------------------------- verified answer bank
def _tok(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", s.lower())) - {
        "the", "a", "an", "is", "of", "in", "to", "do", "you", "how", "what", "say", "dzongkha", "does", "me", "please", "can", "i"}


class AnswerBank:
    """Hand-checked English/Dzongkha question -> Dzongkha answer pairs.
    Anything that matches is returned verbatim, so spelling is exactly what you verified."""

    def __init__(self, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.items = [d for d in data if d.get("dz_answer")]

    def _keys(self, item):
        yield from item.get("questions", [])

    def lookup(self, query: str, threshold: float = 0.6):
        """Exact match first. Otherwise a fuzzy match that is deliberately strict:
        the *content words* must really overlap (so "who is the king of bhutan" can no
        longer match "what is the capital of bhutan" just because both say "bhutan"),
        and numbers must be identical (so "5" never matches "6")."""
        q = normalize(query, digits=False, strip_foreign=False).strip().lower().rstrip("?!. ")
        qt = _tok(q)
        best, best_score = None, 0.0
        for it in self.items:
            for k in self._keys(it):
                kk = normalize(k, digits=False, strip_foreign=False).lower().rstrip("?!. ")
                if q == kk:
                    return it, 1.0
                kt = _tok(kk)
                if qt and kt:                                   # English / Latin text
                    if {t for t in qt if t.isdigit()} != {t for t in kt if t.isdigit()}:
                        continue
                    score = len(qt & kt) / len(qt | kt)         # content-word overlap
                elif not qt and not kt:                         # Dzongkha text
                    score = difflib.SequenceMatcher(None, q, kk).ratio()
                    if score < 0.85:
                        continue
                else:
                    continue
                if score > best_score:
                    best, best_score = it, score
        return (best, best_score) if best_score >= threshold else (None, best_score)

    def build_lexicon(self, lex: Lexicon):
        for it in self.items:
            lex.add_text(it["dz_answer"])
            for q in it.get("questions", []):
                if TIB.search(q):
                    lex.add_text(q)



BHUTAN_TERMS = ("bhutan", "bhutanese", "thimphu", "dzongkha", "druk", "paro", "punakha")

# Other countries (and their adjectives). If the question names one of these and does NOT
# mention Bhutan, the Bhutan facts are skipped and the AI model answers instead.
OTHER_COUNTRIES = {
    "afghanistan", "afghan", "albania", "algeria", "argentina", "argentine", "armenia", "australia", "australian",
    "austria", "austrian", "azerbaijan", "bahrain", "bangladesh", "bangladeshi", "belarus", "belgium", "belgian",
    "bolivia", "bosnia", "brazil", "brazilian", "britain", "british", "england", "english", "scotland", "wales",
    "uk", "united kingdom", "bulgaria", "burma", "cambodia", "cambodian", "cameroon", "canada", "canadian",
    "chile", "china", "chinese", "colombia", "congo", "costa rica", "croatia", "cuba", "cuban", "cyprus",
    "czech", "denmark", "danish", "ecuador", "egypt", "egyptian", "estonia", "ethiopia", "finland", "finnish",
    "france", "french", "georgia", "germany", "german", "ghana", "greece", "greek", "hungary", "iceland",
    "india", "indian", "indonesia", "indonesian", "iran", "iranian", "iraq", "iraqi", "ireland", "irish",
    "israel", "israeli", "italy", "italian", "jamaica", "japan", "japanese", "jordan", "kazakhstan", "kenya",
    "korea", "korean", "kuwait", "laos", "latvia", "lebanon", "libya", "lithuania", "luxembourg", "malaysia",
    "malaysian", "maldives", "mali", "malta", "mexico", "mexican", "mongolia", "morocco", "myanmar", "nepal",
    "nepali", "nepalese", "netherlands", "dutch", "new zealand", "nigeria", "norway", "norwegian", "oman",
    "pakistan", "pakistani", "palestine", "panama", "peru", "philippines", "filipino", "poland", "polish",
    "portugal", "portuguese", "qatar", "romania", "russia", "russian", "rwanda", "saudi", "senegal", "serbia",
    "singapore", "slovakia", "slovenia", "somalia", "south africa", "spain", "spanish", "sri lanka", "sudan",
    "sweden", "swedish", "switzerland", "swiss", "syria", "taiwan", "tajikistan", "tanzania", "thailand", "thai",
    "tibet", "tibetan", "tunisia", "turkey", "turkish", "uganda", "ukraine", "ukrainian", "uae", "emirates",
    "usa", "us", "america", "american", "united states", "uruguay", "uzbekistan", "venezuela", "vietnam",
    "vietnamese", "yemen", "zambia", "zimbabwe", "europe", "european", "africa", "african", "asia", "asian",
    "world",
}


def _mentions(q: str, words) -> bool:
    return any(re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", q) for w in words)


class FactBank:
    """English facts about Bhutan (facts_en.json). Used to GROUND the chatbot: if the question
    is about Bhutan and matches a fact, the answer comes from the fact, not from the language
    model's memory. Questions about any other country are NOT matched (the AI model answers them).
    Each fact: {"id", "keywords": [...], "answer_en": "..."}"""

    def __init__(self, path: str | Path):
        p = Path(path)
        self.items = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    def lookup(self, query: str, max_facts: int = 2):
        q = " " + re.sub(r"[^a-z0-9' ]+", " ", query.lower()) + " "
        if not _mentions(q, BHUTAN_TERMS) and _mentions(q, OTHER_COUNTRIES):
            return []                      # e.g. "capital of India" -> not a Bhutan question
        scored = []
        for f in self.items:
            hits = sum(1 for kw in f.get("keywords", []) if _mentions(q, [kw.lower()]))
            if hits:
                scored.append((hits, f))
        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:max_facts]]

# ------------------------------------------------------------------ God Mode
# "/god" turns on a translate-only mode: every message is translated instead of answered.
#   /god            toggle God Mode          /god on  /god off
#   /god <text>     translate <text> once (God Mode state unchanged)
_GOD = re.compile(r"^\s*/god[\s_-]*(?:mo+de?\b)?\s*(.*)$", re.I | re.S)


def parse_god_command(message: str):
    """None if the message is not a /god command, else (action, text).
    action is 'toggle' | 'on' | 'off' | 'translate' (text is only set for 'translate')."""
    m = _GOD.match(message or "")
    if not m:
        return None
    rest = m.group(1).strip()
    low = rest.lower()
    if not rest:
        return "toggle", ""
    if low in ("on", "start", "enable"):
        return "on", ""
    if low in ("off", "stop", "exit", "end", "disable"):
        return "off", ""
    return "translate", rest


def split_dz_sentences(text: str) -> list[str]:
    """Split Dzongkha text after each shad so long passages are translated sentence by sentence."""
    parts = [p.strip() for p in re.split(f"(?<={SHAD})\\s*", text or "")]
    return [p for p in parts if p and p.strip(SHAD + " ")]