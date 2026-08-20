"""Name anonymization using spaCy NER with regex fallback."""
from __future__ import annotations

import logging
import re
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_PLACEHOLDER = "[STUDENT]"


@lru_cache(maxsize=1)
def _load_nlp(model: str = "en_core_web_sm"):
    try:
        import spacy
        return spacy.load(model)
    except Exception as e:
        logger.warning("spaCy model '%s' unavailable (%s). Regex fallback active.", model, e)
        return None


def anonymize_text(
    text: str,
    placeholder: str = _DEFAULT_PLACEHOLDER,
    spacy_model: str = "en_core_web_sm",
) -> str:
    """Replace detected person names with *placeholder*.

    Uses spaCy PERSON entities when available; falls back to a capitalized
    two-word pattern heuristic otherwise.
    """
    if not text or not text.strip():
        return text

    nlp = _load_nlp(spacy_model)
    if nlp is not None:
        doc = nlp(text)
        result = text
        # Iterate in reverse so earlier character offsets stay valid.
        for ent in reversed(list(doc.ents)):
            if ent.label_ == "PERSON":
                result = result[: ent.start_char] + placeholder + result[ent.end_char :]
        return result

    # Regex fallback: "FirstName LastName" capitalized patterns.
    return re.sub(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b", placeholder, text)


def anonymize_series(
    series: pd.Series,
    placeholder: str = _DEFAULT_PLACEHOLDER,
    spacy_model: str = "en_core_web_sm",
) -> pd.Series:
    """Anonymize a pandas Series of text strings in place (returns new Series).

    Same input/output contract as before (str in, str out, same index,
    empty/whitespace-only strings pass through unchanged). The only change is
    *how* spaCy is invoked: batched via nlp.pipe() instead of once per row via
    nlp(text) inside a Series.apply() loop. Profiling showed the per-row calls
    were the dominant cost (~200 rows/sec) even though PERSON-entity detection
    and replacement logic is identical.
    """
    texts = series.astype(str).tolist()
    nlp = _load_nlp(spacy_model)

    if nlp is None:
        # Regex fallback path — unchanged, no spaCy overhead to batch away.
        return series.astype(str).apply(
            lambda t: anonymize_text(t, placeholder=placeholder, spacy_model=spacy_model)
        )

    # Default to passthrough for every row (matches anonymize_text's early
    # return for empty/whitespace-only text) — only non-blank rows go to
    # nlp.pipe(), same short-circuit anonymize_text does per row.
    results = list(texts)
    to_process = []
    idx_map = []
    for i, t in enumerate(texts):
        if t and t.strip():
            to_process.append(t)
            idx_map.append(i)

    # batch_size=64 is a starting point, not tuned — adjust here if profiling
    # later shows headroom (larger batches vs. memory/latency tradeoff).
    for i, doc in zip(idx_map, nlp.pipe(to_process, batch_size=64)):
        result = texts[i]
        # Iterate in reverse so earlier character offsets stay valid.
        for ent in reversed(list(doc.ents)):
            if ent.label_ == "PERSON":
                result = result[: ent.start_char] + placeholder + result[ent.end_char :]
        results[i] = result

    return pd.Series(results, index=series.index)
