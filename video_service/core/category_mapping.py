import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from video_service.core.utils import logger

ID_COLUMN = "ID"
CATEGORY_COLUMN = "Freewheel Industry Category"
REQUIRED_COLUMNS = (ID_COLUMN, CATEGORY_COLUMN)
DEFAULT_CATEGORY_CSV_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "categories.csv"
).resolve()
UNKNOWN_CATEGORY_VALUES = {"unknown", "none", "n/a", "n-a", ""}
_critical_messages_logged: set[str] = set()
_PRODUCT_CUE_STOPWORDS = {
    "a",
    "ad",
    "advertisement",
    "all",
    "an",
    "and",
    "are",
    "attached",
    "bilingual",
    "but",
    "brand",
    "branding",
    "categories",
    "category",
    "clear",
    "care",
    "clearly",
    "company",
    "confirmation",
    "confirm",
    "confirms",
    "contains",
    "context",
    "corrupted",
    "corner",
    "correct",
    "correctly",
    "designed",
    "despite",
    "evidence",
    "explicit",
    "explicitly",
    "educational",
    "english",
    "enough",
    "final",
    "financial",
    "five",
    "for",
    "frame",
    "fragment",
    "fragments",
    "french",
    "from",
    "further",
    "hair",
    "heavily",
    "image",
    "images",
    "identify",
    "identified",
    "including",
    "its",
    "item",
    "items",
    "label",
    "labels",
    "line",
    "logo",
    "logos",
    "marketing",
    "multiple",
    "not",
    "of",
    "offering",
    "offer",
    "offers",
    "ocr",
    "overall",
    "packaging",
    "parent",
    "point",
    "points",
    "promotion",
    "promotional",
    "promotes",
    "product",
    "products",
    "program",
    "programs",
    "proudly",
    "school",
    "schools",
    "scholarship",
    "scholarships",
    "sponsored",
    "support",
    "supported",
    "reference",
    "references",
    "right",
    "set",
    "setting",
    "service",
    "services",
    "show",
    "showing",
    "shows",
    "shown",
    "studio",
    "text",
    "that",
    "the",
    "their",
    "them",
    "they",
    "these",
    "this",
    "those",
    "top",
    "trade",
    "typo",
    "typos",
    "two",
    "tuition",
    "use",
    "uses",
    "visual",
    "visually",
    "well",
    "well-known",
    "which",
    "with",
    "woman",
    "sufficient",
    "back-to-school",
    "contest",
    "contests",
    "campaign",
    "campaigns",
    "brands",
    "bursary",
    "bursaries",
    "study",
    "back",
    "known",
    "tied",
    "promoting",
}
_PRODUCT_CUE_SINGULARS = {
    "conditioners": "conditioner",
    "formulations": "formulation",
    "shampoos": "shampoo",
    "treatments": "treatment",
    "vitamins": "vitamin",
}
_GENERIC_FAMILY_CONTEXT_TOKENS = {
    "all",
    "and",
    "care",
    "cleaning",
    "consumer",
    "counter",
    "else",
    "entertainment",
    "financial",
    "general",
    "goods",
    "health",
    "healthcare",
    "home",
    "household",
    "internet",
    "manufacture",
    "media",
    "merchandise",
    "over",
    "product",
    "products",
    "retail",
    "sale",
    "service",
    "services",
    "store",
    "stores",
    "telecommunications",
    "technology",
}


def normalize_whitespace(value: str) -> str:
    return " ".join(str(value).split())


def _mapping_text_has_signal(value: str) -> bool:
    tokens = [token for token in normalize_whitespace(value).split() if token]
    if not tokens:
        return False
    return len("".join(tokens)) >= 4


def _looks_generic_freeform_category(value: str) -> bool:
    normalized = normalize_whitespace(value).lower()
    if not normalized or normalized in UNKNOWN_CATEGORY_VALUES:
        return False

    generic_terms = {
        "technology",
        "internet",
        "service",
        "services",
        "consumer",
        "electronics",
        "retail",
        "financial",
        "banking",
        "insurance",
        "healthcare",
        "travel",
        "media",
        "entertainment",
        "telecommunications",
        "telecommunication",
        "educational",
        "education",
        "search",
        "engine",
    }
    tokens = {token for token in normalized.replace("/", " ").split() if token}
    if not tokens:
        return False
    return tokens.issubset(generic_terms)


def _looks_ambiguous_product_family_category(value: str) -> bool:
    normalized = normalize_whitespace(value).lower()
    if not normalized or normalized in UNKNOWN_CATEGORY_VALUES:
        return False

    compact = normalized.replace("/", " ").replace("-", " ")
    product_family_phrases = {
        "hair care",
        "skin care",
        "body care",
        "oral care",
        "personal care",
        "beauty care",
        "pet care",
    }
    return compact in product_family_phrases


def _exact_taxonomy_category_accepts_specificity_hint(value: str) -> bool:
    normalized = normalize_whitespace(value).lower()
    if not normalized or normalized in UNKNOWN_CATEGORY_VALUES:
        return False

    hint_terms = {
        "manufacture",
        "sale",
        "services",
        "service",
        "products",
        "product",
        "store",
        "stores",
        "providers",
        "provider",
        "pharmaceutical",
        "pharmaceuticals",
        "over the counter",
    }
    return any(term in normalized for term in hint_terms)


def _looks_like_product_cue_token(token: str) -> bool:
    compact = re.sub(r"[^A-Za-zÀ-ÿ0-9]+", "", token or "")
    if len(compact) < 2 or compact.isdigit():
        return False

    letters = sum(1 for char in compact if char.isalpha())
    if letters == 0:
        return False

    if len(compact) >= 5:
        vowels = sum(1 for char in compact.lower() if char in "aeiouyàâäéèêëîïôöùûü")
        if vowels == 0:
            return False
        if len(set(compact.lower())) <= 2:
            return False
    return True


def _extract_product_cue_terms(
    text: str,
    *,
    extra_stopwords: set[str] | None = None,
    max_terms: int = 10,
) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    stopwords = set(_PRODUCT_CUE_STOPWORDS)
    if extra_stopwords:
        stopwords.update(token.lower() for token in extra_stopwords if token)

    terms: list[str] = []
    seen: set[str] = set()
    for raw_token in re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9&+/-]*", normalized):
        token = raw_token.strip("&+/-")
        if not token:
            continue

        lower = token.lower()
        lower = _PRODUCT_CUE_SINGULARS.get(lower, lower)
        if lower in stopwords:
            continue
        if len(lower) < 3 and lower not in {"hb"}:
            continue
        if not _looks_like_product_cue_token(lower):
            continue
        if lower in seen:
            continue
        seen.add(lower)
        terms.append(token if lower == token.lower() else token)
        if len(terms) >= max_terms:
            break
    return [(_PRODUCT_CUE_SINGULARS.get(term.lower(), term.lower()) if term.islower() else _PRODUCT_CUE_SINGULARS.get(term.lower(), term)) for term in terms]


def _extract_family_cue_terms(
    family_context: str,
    *,
    max_terms: int = 2,
) -> list[str]:
    normalized = normalize_whitespace(family_context)
    if not normalized:
        return []

    terms: list[str] = []
    seen: set[str] = set()
    for raw_token in re.findall(r"[A-Za-zÀ-ÿ0-9]{3,}", normalized):
        lower = _PRODUCT_CUE_SINGULARS.get(raw_token.lower(), raw_token.lower())
        if lower in _GENERIC_FAMILY_CONTEXT_TOKENS:
            continue
        if not _looks_like_product_cue_token(lower):
            continue
        if lower in seen:
            continue
        seen.add(lower)
        terms.append(lower)
        if len(terms) >= max_terms:
            break
    return terms


def build_product_cue_query_text(
    *,
    predicted_brand: str = "",
    ocr_summary: str = "",
    reasoning_summary: str = "",
    family_context: str = "",
    max_terms: int = 10,
    max_chars: int = 260,
) -> str:
    brand_norm = normalize_whitespace(predicted_brand)
    family_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-zÀ-ÿ0-9]{2,}", normalize_whitespace(family_context))
    }
    brand_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-zÀ-ÿ0-9]{2,}", brand_norm)
    }
    excluded_tokens = family_tokens | brand_tokens

    cue_terms: list[str] = []
    seen: set[str] = set()
    if brand_norm and brand_norm.lower() not in UNKNOWN_CATEGORY_VALUES:
        seen.add(brand_norm.casefold())

    def _extend(tokens: list[str]) -> None:
        for token in tokens:
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            cue_terms.append(token)
            if len(cue_terms) >= max_terms:
                break

    family_terms = _extract_family_cue_terms(family_context, max_terms=max_terms)
    prioritize_family_terms = "/" in family_context or "&" in family_context
    if prioritize_family_terms:
        _extend(family_terms)

    reasoning_terms = _extract_product_cue_terms(
        reasoning_summary,
        extra_stopwords=excluded_tokens,
        max_terms=max_terms,
    )
    _extend(reasoning_terms)

    if len(cue_terms) < 4 and not prioritize_family_terms:
        _extend(family_terms)

    if len(cue_terms) < 4:
        ocr_terms = _extract_product_cue_terms(
            ocr_summary,
            extra_stopwords=excluded_tokens,
            max_terms=max_terms,
        )
        _extend(ocr_terms)

    compact_terms = " ".join(cue_terms[:max_terms]).strip()
    query_parts = [part for part in (brand_norm, compact_terms) if part]
    return normalize_whitespace(" ".join(query_parts))[:max_chars]


def _log_critical_once(message: str) -> None:
    if message in _critical_messages_logged:
        return
    logger.critical(message)
    _critical_messages_logged.add(message)


def select_mapping_input_text(
    raw_category: str,
    suggested_categories_text: str = "",
    predicted_brand: str = "",
    ocr_summary: str = "",
    ocr_max_chars: int = 400,
    exact_taxonomy_match: bool = False,
    reasoning_summary: str = "",
) -> str:
    raw_norm = normalize_whitespace(raw_category)
    brand_norm = normalize_whitespace(predicted_brand)
    ocr_norm = normalize_whitespace(ocr_summary)
    reasoning_norm = normalize_whitespace(reasoning_summary)

    evidence_parts: list[str] = []
    if brand_norm.lower() not in UNKNOWN_CATEGORY_VALUES:
        evidence_parts.append(brand_norm)
    if _mapping_text_has_signal(ocr_norm):
        evidence_parts.append(ocr_norm[:ocr_max_chars])
    evidence_text = "\n".join(evidence_parts)
    ocr_support_text = ocr_norm[:ocr_max_chars] if _mapping_text_has_signal(ocr_norm) else ""
    support_text = reasoning_norm[:ocr_max_chars] if _mapping_text_has_signal(reasoning_norm) else ""

    if raw_norm.lower() not in UNKNOWN_CATEGORY_VALUES and exact_taxonomy_match:
        if (
            _exact_taxonomy_category_accepts_specificity_hint(raw_norm)
        ):
            if evidence_text:
                return f"{raw_norm}\n{evidence_text}"
            if support_text:
                return f"{raw_norm}\n{support_text}"
        return raw_norm

    if (
        raw_norm.lower() not in UNKNOWN_CATEGORY_VALUES
        and not exact_taxonomy_match
        and _looks_generic_freeform_category(raw_norm)
        and evidence_text
    ):
        return evidence_text

    if (
        raw_norm.lower() not in UNKNOWN_CATEGORY_VALUES
        and not exact_taxonomy_match
        and _looks_ambiguous_product_family_category(raw_norm)
    ):
        compact_cues = build_product_cue_query_text(
            predicted_brand=brand_norm,
            ocr_summary=ocr_norm,
            reasoning_summary=reasoning_norm,
            family_context=raw_norm,
            max_chars=ocr_max_chars,
        )
        if compact_cues:
            return f"{raw_norm}\n{compact_cues}"
        if ocr_support_text:
            return f"{raw_norm}\n{ocr_support_text}"
        if support_text:
            return f"{raw_norm}\n{support_text}"

    if raw_norm.lower() not in UNKNOWN_CATEGORY_VALUES:
        return raw_norm

    if brand_norm.lower() not in UNKNOWN_CATEGORY_VALUES:
        return brand_norm

    if ocr_norm:
        return ocr_norm[:ocr_max_chars]

    # Keep mapping non-empty when taxonomy is enabled.
    return "unknown"


@dataclass(frozen=True)
class CategoryMappingState:
    enabled: bool
    category_to_id: Dict[str, str]
    csv_path_used: str
    last_error: Optional[str]

    @property
    def count(self) -> int:
        return len(self.category_to_id)

    def diagnostics(self) -> dict:
        return {
            "category_mapping_enabled": self.enabled,
            "category_mapping_count": self.count,
            "category_csv_path_used": self.csv_path_used,
            "last_error": self.last_error,
        }


def resolve_category_csv_path(csv_path: Optional[str] = None) -> Path:
    env_path = os.environ.get("CATEGORY_CSV_PATH")
    chosen = csv_path or env_path or str(DEFAULT_CATEGORY_CSV_PATH)
    return Path(chosen).expanduser().resolve()


def load_category_mapping(csv_path: Optional[str] = None) -> CategoryMappingState:
    path = resolve_category_csv_path(csv_path)
    path_str = str(path)

    if not path.exists():
        error = f"category mapper disabled: CSV not found at '{path_str}'"
        _log_critical_once(error)
        return CategoryMappingState(
            enabled=False,
            category_to_id={},
            csv_path_used=path_str,
            last_error=error,
        )

    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as exc:
        error = f"category mapper disabled: failed to read CSV at '{path_str}': {exc}"
        _log_critical_once(error)
        return CategoryMappingState(
            enabled=False,
            category_to_id={},
            csv_path_used=path_str,
            last_error=error,
        )

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        error = (
            f"category mapper disabled: missing required columns {missing} "
            f"in '{path_str}' (required={list(REQUIRED_COLUMNS)})"
        )
        _log_critical_once(error)
        return CategoryMappingState(
            enabled=False,
            category_to_id={},
            csv_path_used=path_str,
            last_error=error,
        )

    cat_df = df[[ID_COLUMN, CATEGORY_COLUMN]].copy()
    cat_df[ID_COLUMN] = cat_df[ID_COLUMN].astype(str).map(normalize_whitespace)
    cat_df[CATEGORY_COLUMN] = cat_df[CATEGORY_COLUMN].astype(str).map(normalize_whitespace)
    cat_df = cat_df[(cat_df[ID_COLUMN] != "") & (cat_df[CATEGORY_COLUMN] != "")]

    category_to_id = dict(zip(cat_df[CATEGORY_COLUMN], cat_df[ID_COLUMN]))
    logger.info(
        "category mapper enabled: loaded %d rows from %s",
        len(category_to_id),
        path_str,
    )
    return CategoryMappingState(
        enabled=True,
        category_to_id=category_to_id,
        csv_path_used=path_str,
        last_error=None,
    )


CATEGORY_MAPPING_STATE = load_category_mapping()


def get_category_mapping_diagnostics() -> dict:
    return CATEGORY_MAPPING_STATE.diagnostics()
