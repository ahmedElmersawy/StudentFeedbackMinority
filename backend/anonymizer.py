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
    """Anonymize a pandas Series of text strings in place (returns new Series)."""
    return series.astype(str).apply(
        lambda t: anonymize_text(t, placeholder=placeholder, spacy_model=spacy_model)
    )
