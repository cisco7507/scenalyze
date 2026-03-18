import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from video_service.core.utils import logger

DEFAULT_CATEGORY_JSON_PATH = (
    Path(__file__).resolve().parents[2] / "freewheel.json"
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
    "always",
    "being",
    "bilingual",
    "bottle",
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
    "classic",
    "designed",
    "despite",
    "evidence",
    "explicit",
    "explicitly",
    "educational",
    "english",
    "enough",
    "ensure",
    "final",
    "financial",
    "five",
    "for",
    "frame",
    "frames",
    "fragment",
    "fragments",
    "french",
    "from",
    "further",
    "hair",
    "healthy",
    "heavily",
    "image",
    "images",
    "identify",
    "identified",
    "including",
    "its",
    "itself",
    "item",
    "items",
    "label",
    "labels",
    "labeled",
    "line",
    "logo",
    "logos",
    "marketing",
    "medication",
    "medications",
    "manufacturer",
    "manufacturers",
    "multiple",
    "name",
    "not",
    "of",
    "offering",
    "offer",
    "offers",
    "ocr",
    "otc",
    "overall",
    "packaging",
    "parent",
    "person",
    "point",
    "points",
    "promotion",
    "promotional",
    "promoted",
    "promotes",
    "product",
    "products",
    "program",
    "programs",
    "proudly",
    "read",
    "retail",
    "school",
    "schools",
    "scholarship",
    "scholarships",
    "sponsored",
    "support",
    "supported",
    "reference",
    "references",
    "primary",
    "right",
    "set",
    "setting",
    "service",
    "services",
    "show",
    "showing",
    "shows",
    "shown",
    "slogan",
    "specific",
    "store",
    "stores",
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
    "treat",
    "back",
    "known",
    "tied",
    "promoting",
    "follow",
    "drug",
    "drugstore",
    "drugstores",
    "while",
    "une",
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
_GENERIC_BRANCH_TOKEN_CACHE_KEY: tuple[str, int] | None = None
_GENERIC_BRANCH_TOKEN_CACHE: dict[str, object] | None = None


def _normalize_generic_category_token(token: str) -> str:
    normalized = normalize_whitespace(token or "").lower()
    normalized = re.sub(r"[^a-z0-9]+", "", normalized)
    if not normalized:
        return ""
    if normalized.endswith("ies") and len(normalized) > 4:
        normalized = f"{normalized[:-3]}y"
    elif normalized.endswith("s") and len(normalized) > 4:
        normalized = normalized[:-1]
    return normalized


def _generic_category_tokens(value: str) -> set[str]:
    return {
        token
        for token in (
            _normalize_generic_category_token(raw_token)
            for raw_token in re.findall(r"[A-Za-z0-9]+", normalize_whitespace(value or ""))
        )
        if token
    }


def _get_generic_branch_token_stats() -> dict[str, object]:
    global _GENERIC_BRANCH_TOKEN_CACHE_KEY, _GENERIC_BRANCH_TOKEN_CACHE

    records = tuple(getattr(CATEGORY_MAPPING_STATE, "records", ()) or ())
    cache_key = (str(getattr(CATEGORY_MAPPING_STATE, "json_path_used", "") or ""), len(records))
    if _GENERIC_BRANCH_TOKEN_CACHE_KEY == cache_key and _GENERIC_BRANCH_TOKEN_CACHE is not None:
        return _GENERIC_BRANCH_TOKEN_CACHE

    child_parent_ids = {
        str(record.parent_id or "").strip()
        for record in records
        if str(record.parent_id or "").strip() not in {"", "0"}
    }
    branch_token_sets: list[set[str]] = []
    branch_tokens: set[str] = set()
    branch_token_frequency: dict[str, int] = {}
    for record in records:
        record_id = str(record.category_id or "").strip()
        if record_id not in child_parent_ids and int(getattr(record, "level", 0) or 0) > 1:
            continue
        tokens = _generic_category_tokens(getattr(record, "name", ""))
        if not tokens:
            continue
        branch_token_sets.append(tokens)
        branch_tokens.update(tokens)
        for token in tokens:
            branch_token_frequency[token] = int(branch_token_frequency.get(token, 0)) + 1

    _GENERIC_BRANCH_TOKEN_CACHE_KEY = cache_key
    _GENERIC_BRANCH_TOKEN_CACHE = {
        "branch_token_sets": tuple(branch_token_sets),
        "branch_tokens": branch_tokens,
        "branch_token_frequency": branch_token_frequency,
    }
    return _GENERIC_BRANCH_TOKEN_CACHE


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
        "food",
        "beverage",
        "beverages",
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
    tokens = _generic_category_tokens(normalized.replace("/", " "))
    if not tokens:
        return False
    if tokens.issubset(generic_terms):
        return True

    branch_stats = _get_generic_branch_token_stats()
    branch_token_sets = tuple(branch_stats.get("branch_token_sets", ()) or ())
    branch_tokens = set(branch_stats.get("branch_tokens", set()) or set())
    branch_token_frequency = dict(branch_stats.get("branch_token_frequency", {}) or {})
    if any(tokens == branch_tokens_candidate for branch_tokens_candidate in branch_token_sets):
        return True
    if len(tokens) <= 2 and tokens.issubset(branch_tokens) and all(
        int(branch_token_frequency.get(token, 0)) > 1 for token in tokens
    ):
        return True
    return False


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

    broad_exact_labels = {
        "pharmaceutical manufacture and sale - over the counter",
        "pharmaceutical manufacture and sale - prescription",
        "pharmaceutical manufacture and sale",
        "telecommunication services - all else",
        "retail and general merchandise - all else",
        "haircare products - all else",
        "home products - all else",
    }
    if normalized in broad_exact_labels:
        return True

    return normalized.endswith("- all else")


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
            compact_cues = build_product_cue_query_text(
                predicted_brand=brand_norm,
                ocr_summary=ocr_norm,
                reasoning_summary=reasoning_norm,
                family_context=raw_norm,
                max_chars=ocr_max_chars,
            )
            if compact_cues and compact_cues.casefold() != brand_norm.casefold():
                return compact_cues
            if evidence_text:
                return f"{raw_norm}\n{evidence_text}"
            if support_text:
                return f"{raw_norm}\n{support_text}"
        return raw_norm

    if (
        raw_norm.lower() not in UNKNOWN_CATEGORY_VALUES
        and not exact_taxonomy_match
        and _looks_generic_freeform_category(raw_norm)
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
        return raw_norm

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
        return raw_norm

    if raw_norm.lower() not in UNKNOWN_CATEGORY_VALUES:
        return raw_norm

    if brand_norm.lower() not in UNKNOWN_CATEGORY_VALUES:
        return brand_norm

    if ocr_norm:
        return ocr_norm[:ocr_max_chars]

    # Keep mapping non-empty when taxonomy is enabled.
    return "unknown"


@dataclass(frozen=True)
class CategoryTaxonomyRecord:
    category_id: str
    name: str
    parent_id: str
    parent_name: str
    level: int
    path_ids: tuple[str, ...]
    path_names: tuple[str, ...]
    path_text: str
    industry_id: str
    industry_name: str


@dataclass(frozen=True)
class CategoryMappingState:
    enabled: bool
    category_to_id: Dict[str, str]
    category_to_industry_id: Dict[str, str]
    category_to_industry_name: Dict[str, str]
    category_to_parent_id: Dict[str, str]
    category_to_parent: Dict[str, str]
    category_to_path_text: Dict[str, str]
    category_to_level: Dict[str, int]
    records: tuple[CategoryTaxonomyRecord, ...]
    json_path_used: str
    last_error: Optional[str]

    @property
    def count(self) -> int:
        return len(self.category_to_id)

    def diagnostics(self) -> dict:
        return {
            "category_mapping_enabled": self.enabled,
            "category_mapping_count": self.count,
            "category_json_path_used": self.json_path_used,
            "last_error": self.last_error,
        }


@dataclass(frozen=True)
class CategoryExplorerGroupChild:
    id: str
    name: str


@dataclass(frozen=True)
class CategoryExplorerGroup:
    id: str
    name: str
    children: tuple[CategoryExplorerGroupChild, ...]


@dataclass(frozen=True)
class CategoryExplorerItem:
    id: str
    name: str
    level: int
    parent_id: str
    path_ids: tuple[str, ...]
    path_names: tuple[str, ...]
    path_text: str
    industry_id: str
    industry_name: str


@dataclass(frozen=True)
class CategoryExplorerState:
    enabled: bool
    groups: tuple[CategoryExplorerGroup, ...]
    items: tuple[CategoryExplorerItem, ...]
    json_path_used: str
    last_error: Optional[str]

    def diagnostics(self) -> dict:
        child_parent_ids = {
            str(item.parent_id or "").strip()
            for item in self.items
            if str(item.parent_id or "").strip() not in {"", "0"}
        }
        root_count = sum(1 for item in self.items if str(item.parent_id or "").strip() in {"", "0"})
        leaf_count = sum(1 for item in self.items if str(item.id or "").strip() not in child_parent_ids)
        max_level = max((int(item.level or 0) for item in self.items), default=0)
        return {
            "taxonomy_explorer_enabled": self.enabled,
            "group_count": len(self.groups),
            "item_count": len(self.items),
            "root_count": root_count,
            "leaf_count": leaf_count,
            "max_level": max_level,
            "json_path_used": self.json_path_used,
            "last_error": self.last_error,
        }


def resolve_category_json_path(json_path: Optional[str] = None) -> Path:
    env_path = os.environ.get("CATEGORY_JSON_PATH")
    chosen = json_path or env_path or str(DEFAULT_CATEGORY_JSON_PATH)
    return Path(chosen).expanduser().resolve()


def _normalize_category_id(value: object) -> str:
    return normalize_whitespace(str(value or ""))


def _normalize_parent_id(value: object) -> str:
    normalized = _normalize_category_id(value)
    return normalized or "0"


def _normalize_category_level(value: object) -> int:
    try:
        return int(str(value or "0").strip() or "0")
    except (TypeError, ValueError):
        return 0


def _build_taxonomy_path(
    item_id: str,
    item_lookup: dict[str, dict[str, object]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    path_ids: list[str] = []
    path_names: list[str] = []
    current_id = item_id
    seen: set[str] = set()

    while current_id and current_id != "0" and current_id in item_lookup:
        if current_id in seen:
            raise ValueError(f"Cycle detected in taxonomy at item_id={current_id}")
        seen.add(current_id)
        current_item = item_lookup[current_id]
        path_ids.append(current_id)
        current_name = normalize_whitespace(str(current_item.get("name") or ""))
        if current_name:
            path_names.append(current_name)
        current_id = _normalize_parent_id(current_item.get("parent_id"))

    path_ids.reverse()
    path_names.reverse()
    return tuple(path_ids), tuple(path_names)


def load_category_mapping(json_path: Optional[str] = None) -> CategoryMappingState:
    path = resolve_category_json_path(json_path)
    path_str = str(path)

    if not path.exists():
        error = f"category mapper disabled: taxonomy JSON not found at '{path_str}'"
        _log_critical_once(error)
        return CategoryMappingState(
            enabled=False,
            category_to_id={},
            category_to_industry_id={},
            category_to_industry_name={},
            category_to_parent_id={},
            category_to_parent={},
            category_to_path_text={},
            category_to_level={},
            records=(),
            json_path_used=path_str,
            last_error=error,
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        error = f"category mapper disabled: failed to read taxonomy JSON at '{path_str}': {exc}"
        _log_critical_once(error)
        return CategoryMappingState(
            enabled=False,
            category_to_id={},
            category_to_industry_id={},
            category_to_industry_name={},
            category_to_parent_id={},
            category_to_parent={},
            category_to_path_text={},
            category_to_level={},
            records=(),
            json_path_used=path_str,
            last_error=error,
        )

    items = payload.get("items")
    if not isinstance(items, list) or not items:
        error = f"category mapper disabled: missing non-empty 'items' array in '{path_str}'"
        _log_critical_once(error)
        return CategoryMappingState(
            enabled=False,
            category_to_id={},
            category_to_industry_id={},
            category_to_industry_name={},
            category_to_parent_id={},
            category_to_parent={},
            category_to_path_text={},
            category_to_level={},
            records=(),
            json_path_used=path_str,
            last_error=error,
        )

    item_lookup: dict[str, dict[str, object]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        category_id = _normalize_category_id(item.get("id"))
        category_name = normalize_whitespace(str(item.get("name") or ""))
        if not category_id or not category_name:
            continue
        normalized_item = {
            "id": category_id,
            "name": category_name,
            "level": _normalize_category_level(item.get("level")),
            "parent_id": _normalize_parent_id(item.get("parent_id")),
        }
        item_lookup[category_id] = normalized_item

    if not item_lookup:
        error = f"category mapper disabled: no valid taxonomy items found in '{path_str}'"
        _log_critical_once(error)
        return CategoryMappingState(
            enabled=False,
            category_to_id={},
            category_to_industry_id={},
            category_to_industry_name={},
            category_to_parent_id={},
            category_to_parent={},
            category_to_path_text={},
            category_to_level={},
            records=(),
            json_path_used=path_str,
            last_error=error,
        )

    records: list[CategoryTaxonomyRecord] = []
    category_to_id: dict[str, str] = {}
    category_to_industry_id: dict[str, str] = {}
    category_to_industry_name: dict[str, str] = {}
    category_to_parent_id: dict[str, str] = {}
    category_to_parent: dict[str, str] = {}
    category_to_path_text: dict[str, str] = {}
    category_to_level: dict[str, int] = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        category_id = _normalize_category_id(item.get("id"))
        if not category_id or category_id not in item_lookup:
            continue
        normalized_item = item_lookup[category_id]
        category_name = str(normalized_item["name"])
        if category_name in category_to_id:
            error = (
                f"category mapper disabled: duplicate category name {category_name!r} "
                f"found in '{path_str}'"
            )
            _log_critical_once(error)
            return CategoryMappingState(
                enabled=False,
                category_to_id={},
                category_to_industry_id={},
                category_to_industry_name={},
                category_to_parent_id={},
                category_to_parent={},
                category_to_path_text={},
                category_to_level={},
                records=(),
                json_path_used=path_str,
                last_error=error,
            )

        path_ids, path_names = _build_taxonomy_path(category_id, item_lookup)
        parent_id = str(normalized_item["parent_id"])
        parent_item = item_lookup.get(parent_id)
        parent_name = str(parent_item["name"]) if parent_item else ""
        industry_id = path_ids[0] if path_ids else category_id
        industry_name = path_names[0] if path_names else category_name
        record = CategoryTaxonomyRecord(
            category_id=category_id,
            name=category_name,
            parent_id=parent_id,
            parent_name=parent_name,
            level=int(normalized_item["level"]),
            path_ids=path_ids,
            path_names=path_names,
            path_text=" : ".join(path_names) if path_names else category_name,
            industry_id=industry_id,
            industry_name=industry_name,
        )
        records.append(record)
        category_to_id[record.name] = record.category_id
        category_to_industry_id[record.name] = record.industry_id
        category_to_industry_name[record.name] = record.industry_name
        category_to_parent_id[record.name] = record.parent_id if record.parent_id != "0" else ""
        category_to_parent[record.name] = record.parent_name
        category_to_path_text[record.name] = record.path_text
        category_to_level[record.name] = record.level

    logger.info(
        "category mapper enabled: loaded %d taxonomy items from %s",
        len(records),
        path_str,
    )
    return CategoryMappingState(
        enabled=True,
        category_to_id=category_to_id,
        category_to_industry_id=category_to_industry_id,
        category_to_industry_name=category_to_industry_name,
        category_to_parent_id=category_to_parent_id,
        category_to_parent=category_to_parent,
        category_to_path_text=category_to_path_text,
        category_to_level=category_to_level,
        records=tuple(records),
        json_path_used=path_str,
        last_error=None,
    )


CATEGORY_MAPPING_STATE = load_category_mapping()


def load_category_explorer_state(json_path: Optional[str] = None) -> CategoryExplorerState:
    path = resolve_category_json_path(json_path)
    path_str = str(path)

    if not path.exists():
        error = f"taxonomy explorer disabled: taxonomy JSON not found at '{path_str}'"
        _log_critical_once(error)
        return CategoryExplorerState(
            enabled=False,
            groups=(),
            items=(),
            json_path_used=path_str,
            last_error=error,
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        error = f"taxonomy explorer disabled: failed to read taxonomy JSON at '{path_str}': {exc}"
        _log_critical_once(error)
        return CategoryExplorerState(
            enabled=False,
            groups=(),
            items=(),
            json_path_used=path_str,
            last_error=error,
        )

    items = payload.get("items")
    if not isinstance(items, list) or not items:
        error = f"taxonomy explorer disabled: missing non-empty 'items' array in '{path_str}'"
        _log_critical_once(error)
        return CategoryExplorerState(
            enabled=False,
            groups=(),
            items=(),
            json_path_used=path_str,
            last_error=error,
        )

    item_lookup: dict[str, dict[str, object]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        category_id = _normalize_category_id(item.get("id"))
        category_name = normalize_whitespace(str(item.get("name") or ""))
        if not category_id or not category_name:
            continue
        item_lookup[category_id] = {
            "id": category_id,
            "name": category_name,
            "level": _normalize_category_level(item.get("level")),
            "parent_id": _normalize_parent_id(item.get("parent_id")),
        }

    if not item_lookup:
        error = f"taxonomy explorer disabled: no valid taxonomy items found in '{path_str}'"
        _log_critical_once(error)
        return CategoryExplorerState(
            enabled=False,
            groups=(),
            items=(),
            json_path_used=path_str,
            last_error=error,
        )

    normalized_items: list[CategoryExplorerItem] = []
    for category_id, normalized_item in item_lookup.items():
        path_ids, path_names = _build_taxonomy_path(category_id, item_lookup)
        category_name = str(normalized_item["name"])
        industry_id = path_ids[0] if path_ids else category_id
        industry_name = path_names[0] if path_names else category_name
        normalized_items.append(
            CategoryExplorerItem(
                id=category_id,
                name=category_name,
                level=int(normalized_item["level"]),
                parent_id=str(normalized_item["parent_id"]),
                path_ids=path_ids,
                path_names=path_names,
                path_text=" : ".join(path_names) if path_names else category_name,
                industry_id=industry_id,
                industry_name=industry_name,
            )
        )

    groups_payload = payload.get("groups")
    normalized_groups: list[CategoryExplorerGroup] = []
    if isinstance(groups_payload, list):
        for group in groups_payload:
            if not isinstance(group, dict):
                continue
            group_id = _normalize_category_id(group.get("id"))
            group_name = normalize_whitespace(str(group.get("name") or ""))
            if not group_name:
                continue
            children_payload = group.get("children")
            children: list[CategoryExplorerGroupChild] = []
            if isinstance(children_payload, list):
                for child in children_payload:
                    if not isinstance(child, dict):
                        continue
                    child_id = _normalize_category_id(child.get("id"))
                    child_name = normalize_whitespace(str(child.get("name") or ""))
                    if not child_id or not child_name:
                        continue
                    children.append(CategoryExplorerGroupChild(id=child_id, name=child_name))
            normalized_groups.append(
                CategoryExplorerGroup(
                    id=group_id,
                    name=group_name,
                    children=tuple(children),
                )
            )

    return CategoryExplorerState(
        enabled=True,
        groups=tuple(normalized_groups),
        items=tuple(sorted(normalized_items, key=lambda item: (item.level, item.path_text.casefold(), item.id))),
        json_path_used=path_str,
        last_error=None,
    )


CATEGORY_EXPLORER_STATE = load_category_explorer_state()


def get_category_mapping_diagnostics() -> dict:
    return CATEGORY_MAPPING_STATE.diagnostics()


def get_category_explorer_payload() -> dict:
    diagnostics = CATEGORY_EXPLORER_STATE.diagnostics()
    return {
        "enabled": CATEGORY_EXPLORER_STATE.enabled,
        "json_path_used": CATEGORY_EXPLORER_STATE.json_path_used,
        "last_error": CATEGORY_EXPLORER_STATE.last_error,
        "stats": {
            "group_count": diagnostics["group_count"],
            "item_count": diagnostics["item_count"],
            "root_count": diagnostics["root_count"],
            "leaf_count": diagnostics["leaf_count"],
            "max_level": diagnostics["max_level"],
        },
        "groups": [
            {
                "id": group.id,
                "name": group.name,
                "children": [
                    {
                        "id": child.id,
                        "name": child.name,
                    }
                    for child in group.children
                ],
            }
            for group in CATEGORY_EXPLORER_STATE.groups
        ],
        "items": [
            {
                "id": item.id,
                "name": item.name,
                "level": item.level,
                "parent_id": item.parent_id,
                "path_ids": list(item.path_ids),
                "path_names": list(item.path_names),
                "path_text": item.path_text,
                "industry_id": item.industry_id,
                "industry_name": item.industry_name,
            }
            for item in CATEGORY_EXPLORER_STATE.items
        ],
    }
