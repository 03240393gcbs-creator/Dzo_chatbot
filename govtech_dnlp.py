"""Optional Dzongkha NLP integration helpers.

GovTech's public Dzongkha NLP website documents a hosted NLLB translation
system trained on an approximately 700,000-sentence DCDD parallel corpus.
It does not publish a downloadable corpus or a documented public API contract.
This module therefore does NOT scrape the website or pretend that the private
DCDD corpus is bundled. It integrates the public Dzoseg tokenizer and provides
safe import/validation helpers for a corpus file that the user is authorised to use.

Public tokenizer: https://github.com/KarmaCST/dzoseg
GovTech information: https://dnlp.tech.gov.bt/about
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

TIBETAN_RE = re.compile(r"[\u0F00-\u0FFF]")


def dzongkha_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(TIBETAN_RE.fullmatch(c)) for c in letters) / len(letters)


def segment_dzongkha(text: str, method: str = "tseg") -> list[str]:
    """Segment Dzongkha with Dzoseg when installed; safe Tseg fallback otherwise."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        from dzoseg import DzongkhaTokenizer
        tokenizer = _get_tokenizer()
        if method == "subword":
            return list(tokenizer.segment_subwords(text))
        return list(tokenizer.segment_tseg(text))
    except (ImportError, Exception):
        # Tseg is the traditional syllable separator. This fallback keeps the
        # application usable offline and does not require a model download.
        return [x for x in re.split(r"[་།\s]+", text) if x]


_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from dzoseg import DzongkhaTokenizer
        _TOKENIZER = DzongkhaTokenizer()
    return _TOKENIZER


def read_parallel_pairs(path: str | Path, en_col: str = "english", dz_col: str = "dzongkha") -> list[tuple[str, str]]:
    """Read authorised English/Dzongkha pairs from CSV, TSV, JSON, JSONL or XLSX.

    JSON rows may use english/dzongkha, en/dz, source/target, or instruction/output.
    Text files and unsupported formats are rejected rather than guessed.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    rows: list[dict] = []
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with p.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f, delimiter=delimiter))
    elif suffix in {".json", ".jsonl"}:
        raw = p.read_text(encoding="utf-8-sig")
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            data = json.loads(raw)
            rows = data if isinstance(data, list) else data.get("data", data.get("pairs", []))
    elif suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Install pandas and openpyxl to read Excel files: pip install pandas openpyxl") from exc
        rows = pd.read_excel(p).fillna("").to_dict(orient="records")
    else:
        raise ValueError("Supported corpus formats: .csv, .tsv, .json, .jsonl, .xlsx")

    aliases = ((en_col, dz_col), ("english", "dzongkha"), ("en", "dz"),
               ("source", "target"), ("instruction", "output"))
    pairs: list[tuple[str, str]] = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        pair = None
        for left, right in aliases:
            if row.get(left) and row.get(right):
                pair = (str(row[left]).strip(), str(row[right]).strip())
                break
        if not pair:
            continue
        en, dz = pair
        if len(en) < 2 or len(dz) < 2 or dzongkha_ratio(dz) < 0.5:
            continue
        if dzongkha_ratio(en) > 0.2:
            continue
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def export_training_jsonl(pairs: Iterable[tuple[str, str]], output: str | Path) -> int:
    """Write cleaned pairs as JSONL accepted by the training-data workflow."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for en, dz in pairs:
            record = {"english": en, "dzongkha": dz}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
