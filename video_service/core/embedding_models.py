import os
import torch


SAFE_DEFAULT_CATEGORY_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

SUPPORTED_CATEGORY_EMBEDDING_MODELS = (
    "BAAI/bge-large-en-v1.5",
    "jinaai/jina-embeddings-v3",
    "Alibaba-NLP/gte-large-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "google/embeddinggemma-300m",
)

TRUST_REMOTE_CODE_CATEGORY_EMBEDDING_MODELS = {
    "jinaai/jina-embeddings-v3",
    "Alibaba-NLP/gte-large-en-v1.5",
}

MODEL_ALLOWED_DEVICE_BACKENDS = {
    "BAAI/bge-large-en-v1.5": ("cuda", "mps", "cpu"),
    "sentence-transformers/all-mpnet-base-v2": ("cuda", "mps", "cpu"),
    "sentence-transformers/all-MiniLM-L6-v2": ("cuda", "mps", "cpu"),
    "google/embeddinggemma-300m": ("cuda", "mps", "cpu"),
    "jinaai/jina-embeddings-v3": ("cuda", "cpu"),
    "Alibaba-NLP/gte-large-en-v1.5": ("cuda", "cpu"),
}


def is_supported_category_embedding_model(model_name: str | None) -> bool:
    return str(model_name or "").strip() in SUPPORTED_CATEGORY_EMBEDDING_MODELS


def resolve_category_embedding_model(model_name: str | None) -> str:
    requested = str(model_name or "").strip()
    if requested in SUPPORTED_CATEGORY_EMBEDDING_MODELS:
        return requested
    return SAFE_DEFAULT_CATEGORY_EMBEDDING_MODEL


def category_embedding_model_requires_remote_code(model_name: str | None) -> bool:
    return (
        str(model_name or "").strip()
        in TRUST_REMOTE_CODE_CATEGORY_EMBEDDING_MODELS
    )


def _is_backend_available(backend: str) -> bool:
    if backend == "cuda":
        return bool(torch.cuda.is_available())
    if backend == "mps":
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return backend == "cpu"


def resolve_category_embedding_device(
    model_name: str | None,
    *,
    preferred_device: str | None = None,
) -> str:
    model = resolve_category_embedding_model(model_name)
    allowed = MODEL_ALLOWED_DEVICE_BACKENDS.get(model, ("cpu",))
    preferred = str(preferred_device or "").strip().lower()

    ordered_candidates: list[str] = []
    if preferred:
        ordered_candidates.append(preferred)
    for backend in ("cuda", "mps", "cpu"):
        if backend not in ordered_candidates:
            ordered_candidates.append(backend)

    for backend in ordered_candidates:
        if backend in allowed and _is_backend_available(backend):
            return backend
    return "cpu"


DEFAULT_CATEGORY_EMBEDDING_MODEL = resolve_category_embedding_model(
    os.environ.get("CATEGORY_EMBEDDING_MODEL")
)
