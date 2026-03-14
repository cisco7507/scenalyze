from pathlib import Path

import pytest

from video_service.core.category_mapping import (
    build_product_cue_query_text,
    load_category_mapping,
    select_mapping_input_text,
)
from video_service.core.categories import (
    _prepare_query_text_for_embedding,
    _split_embedding_query_fragments,
    _translate_embedding_fragment_to_english,
    _summarize_mapping_query_for_log,
)
from video_service.core.embedding_models import (
    category_embedding_model_requires_remote_code,
    resolve_category_embedding_device,
    resolve_category_embedding_model,
)

pytestmark = pytest.mark.unit


def test_load_category_mapping_uses_explicit_columns_and_normalizes_whitespace(tmp_path: Path):
    csv_path = tmp_path / "categories.csv"
    csv_path.write_text(
        "ID,Freewheel Industry Category\n"
        " 1 ,  Agriculture   Crop Production  \n"
        "2,Agriculture Livestock Production\n",
        encoding="utf-8",
    )

    mapping_state = load_category_mapping(str(csv_path))

    assert mapping_state.enabled is True
    assert mapping_state.count == 2
    assert mapping_state.category_to_id["Agriculture Crop Production"] == "1"
    assert mapping_state.category_to_id["Agriculture Livestock Production"] == "2"
    assert mapping_state.last_error is None


def test_load_category_mapping_disables_when_required_columns_missing(tmp_path: Path):
    csv_path = tmp_path / "bad_categories.csv"
    csv_path.write_text(
        "WrongID,WrongCategory\n1,Anything\n",
        encoding="utf-8",
    )

    mapping_state = load_category_mapping(str(csv_path))

    assert mapping_state.enabled is False
    assert mapping_state.count == 0
    assert mapping_state.last_error is not None
    assert "missing required columns" in mapping_state.last_error


def test_select_mapping_input_text_fallback_order_without_suggested_categories_bias():
    assert (
        select_mapping_input_text(
            raw_category="Unknown",
            suggested_categories_text="A, B, C",
            predicted_brand="BrandX",
            ocr_summary="OCR text",
        )
        == "BrandX"
    )
    assert (
        select_mapping_input_text(
            raw_category="none",
            suggested_categories_text="",
            predicted_brand="BrandX",
            ocr_summary="OCR text",
        )
        == "BrandX"
    )
    assert (
        select_mapping_input_text(
            raw_category="n/a",
            suggested_categories_text="",
            predicted_brand="unknown",
            ocr_summary="  Long OCR summary text  ",
        )
        == "Long OCR summary text"
    )


def test_select_mapping_input_text_uses_compact_cues_for_generic_freeform_category():
    assert (
        select_mapping_input_text(
            raw_category="Technology / Internet Services",
            predicted_brand="Google",
            ocr_summary="Google Pixel 9",
            reasoning_summary="Google Pixel 9 smartphone mobile device",
            exact_taxonomy_match=False,
        )
        == "Technology / Internet Services\nGoogle Pixel smartphone mobile device"
    )


def test_select_mapping_input_text_preserves_exact_taxonomy_match():
    assert (
        select_mapping_input_text(
            raw_category="Language Learning",
            predicted_brand="OnParle",
            ocr_summary="French lessons",
            exact_taxonomy_match=True,
        )
        == "Language Learning"
    )


def test_select_mapping_input_text_preserves_specific_exact_taxonomy_leaf():
    assert (
        select_mapping_input_text(
            raw_category="Grocery Stores",
            predicted_brand="No Frills",
            ocr_summary="NOFRILLS Loblaws Inc Prices in effect until February 2026 YES SAVINGS",
            reasoning_summary=(
                "The OCR evidence explicitly references NOFRILLS and Loblaws Inc., "
                "confirming it is a promotional ad for the No Frills grocery chain."
            ),
            exact_taxonomy_match=True,
        )
        == "Grocery Stores"
    )


def test_select_mapping_input_text_keeps_specific_freeform_category():
    assert (
        select_mapping_input_text(
            raw_category="Consumer Goods / Household Cleaning & Laundry",
            predicted_brand="Tide & Downy",
            ocr_summary="Tide Downy laundry detergent",
            exact_taxonomy_match=False,
        )
        == "Consumer Goods / Household Cleaning & Laundry"
    )


def test_select_mapping_input_text_appends_reasoning_for_ambiguous_product_family():
    assert (
        select_mapping_input_text(
            raw_category="Hair Care",
            predicted_brand="P&G",
            ocr_summary="HB hair biology volumizing shampoo conditioner thickening treatment",
            reasoning_summary="volumizing shampoo conditioner thickening treatment",
            exact_taxonomy_match=False,
        )
        == "Hair Care\nP&G volumizing shampoo conditioner thickening treatment"
    )


def test_select_mapping_input_text_appends_reasoning_for_broad_exact_taxonomy():
    assert (
        select_mapping_input_text(
            raw_category="Pharmaceutical Manufacture and Sale - over the counter",
            predicted_brand="Pepto-Bismol",
            ocr_summary="upset stomach indigestion nausea diarrhea",
            reasoning_summary="upset stomach indigestion nausea diarrhea",
            exact_taxonomy_match=True,
        )
        == "Pepto-Bismol upset stomach indigestion nausea diarrhea"
    )


def test_select_mapping_input_text_compacts_otc_disclaimer_noise():
    result = select_mapping_input_text(
        raw_category="Pharmaceutical Manufacture and Sale - over the counter",
        predicted_brand="Vicks",
        ocr_summary=(
            "VICKS VapoCOOL VAPORIZE MAX Honey Lemon Chill SORE THROAT PAIN. "
            "miel Citron GLACIAL TO ENSURE THIS PRODUCT IS RIGHT FOR YOU, "
            "ALWAYS READ AND FOLLOW THE LABEL"
        ),
        reasoning_summary="The product shown is Vicks VapoCOOL Vaporize Max Honey Lemon for sore throat pain.",
        exact_taxonomy_match=True,
    )

    assert result == "Vicks VapoCOOL Vaporize Max Honey Lemon sore throat pain"


def test_build_product_cue_query_text_prefers_compact_reasoning_over_noisy_ocr():
    noisy_ocr = (
        "PeG PANIENI miAc PANTF Kescue Fonditionyv REviiai Ein Repakateun "
        "PANTENE Kepaip MIRACLE WTAMNe Conditionek BIOTIN COLLAGEN KERATIN"
    )
    reasoning = (
        "The OCR text contains multiple fragments that clearly reference Pantene and "
        "its product line, including Miracle Rescue, Biotin + Collagen, and "
        "Keratin + Vitamin E. The ad shows shampoos and conditioners."
    )

    assert (
        build_product_cue_query_text(
            predicted_brand="Pantene",
            ocr_summary=noisy_ocr,
            reasoning_summary=reasoning,
            family_context="Hair Care",
        )
        == "Pantene Miracle Rescue Biotin Collagen Keratin Vitamin shampoo conditioner"
    )


def test_build_product_cue_query_text_drops_marketing_language_for_laundry_family():
    reasoning = (
        "The OCR text and attached images clearly show a promotional contest for Tide and "
        "Downy, two well-known household cleaning and laundry brands. The ad is promoting "
        "a back-to-school contest offering study bursaries, but the products shown are "
        "laundry detergent and fabric softener."
    )

    assert (
        build_product_cue_query_text(
            predicted_brand="Tide & Downy",
            ocr_summary="Tfde Tide n fto PaBLer unstopables",
            reasoning_summary=reasoning,
            family_context="Consumer Goods / Household Cleaning & Laundry",
        )
        == "Tide & Downy laundry detergent fabric softener"
    )


def test_build_product_cue_query_text_drops_otc_reasoning_narration():
    reasoning = (
        "The ad clearly promotes a Vicks OTC product (VapoCOOL VAPORIZE MAX) designed "
        "to treat sore throat pain, which is a classic over-the-counter medication. "
        "While Drug Stores and Drugstores are retail categories, the ad is promoting "
        "the manufacturer's product, not the store itself."
    )

    assert (
        build_product_cue_query_text(
            predicted_brand="Vicks",
            ocr_summary="VICKS OTC VapoCOOL VAPORIZE MAX treat sore throat pain classic over-the-counter",
            reasoning_summary=reasoning,
            family_context="Pharmaceutical Manufacture and Sale - over the counter",
        )
        == "Vicks VapoCOOL VAPORIZE MAX sore throat pain over-the-counter"
    )


def test_build_product_cue_query_text_drops_meta_reasoning_tokens_for_haircare():
    reasoning = (
        "The OCR text and visual frames clearly show the product: a bottle labeled head "
        "& shoulders with the specific product name BARE and the French slogan Une "
        "protection antipelliculaire (anti-dandruff protection). The visual of a person "
        "with healthy hair and the product packaging confirm this is a hair care product. "
        "The category is Hair Care as the primary product being promoted is a shampoo or "
        "conditioner for dandruff control."
    )

    assert (
        build_product_cue_query_text(
            predicted_brand="Head & Shoulders",
            ocr_summary="head & shoulders BARE Une protection antipelliculaire",
            reasoning_summary=reasoning,
            family_context="Hair Care",
        )
        == "Head & Shoulders BARE protection antipelliculaire anti-dandruff shampoo conditioner dandruff control"
    )


def test_prepare_query_text_for_embedding_keeps_short_label_style_queries_raw():
    raw_query = "Over-the-Counter Medication"

    assert _prepare_query_text_for_embedding(
        raw_query,
        "BAAI/bge-large-en-v1.5",
    ) == raw_query
    assert _prepare_query_text_for_embedding(
        raw_query,
        "sentence-transformers/all-mpnet-base-v2",
    ) == raw_query


def test_prepare_query_text_for_embedding_prefixes_long_multiline_bge_queries():
    enriched_query = (
        "Hair Care\n"
        "Head & Shoulders frames bottle labeled specific name BARE slogan "
        "Une protection antipelliculaire"
    )

    assert _prepare_query_text_for_embedding(
        enriched_query,
        "BAAI/bge-large-en-v1.5",
    ) == (
        "Represent this sentence for searching relevant passages: "
        + enriched_query
    )
    assert _prepare_query_text_for_embedding(
        enriched_query,
        "sentence-transformers/all-mpnet-base-v2",
    ) == enriched_query


def test_translate_embedding_fragment_to_english_normalizes_common_french_terms():
    raw_fragment = "Tangerine Changoz dèro bancairo La banque officielle des Raptors de Toronto"

    assert _translate_embedding_fragment_to_english(raw_fragment) == (
        "Tangerine Changoz bank official Raptors Toronto"
    )


def test_split_embedding_query_fragments_includes_raw_label_and_translated_candidates():
    fragments = _split_embedding_query_fragments(
        "Banking",
        "Tangerine\nTangerine Changoz dèro bancairo La banque officielle des Raptors de Toronto. Et de leurs fans.",
    )

    assert fragments == [
        "Banking",
        "Tangerine",
        "Tangerine Changoz bank official Raptors Toronto their fans",
    ]


def test_summarize_mapping_query_for_log_normalizes_and_truncates():
    long_query = (
        "Tangerine\n"
        "Tangerine Tangerine Changoz dèro bancairo La banque officielle des Raptors de Toronto. "
        "Et de leurs fans. Tangerine Changoz dàro bancairo La banque officielle des Raptors de Toronto. "
        "Et de leurs fans."
    )

    summarized = _summarize_mapping_query_for_log(long_query)

    assert "\n" not in summarized
    assert summarized.startswith("Tangerine Tangerine Tangerine Changoz dèro bancairo")
    assert summarized.endswith("...")
    assert len(summarized) <= 180


def test_category_embedding_model_allowlist_resolution():
    assert resolve_category_embedding_model("google/embeddinggemma-300m") == "google/embeddinggemma-300m"
    assert (
        resolve_category_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert resolve_category_embedding_model("not/a-real-model") == "BAAI/bge-large-en-v1.5"


def test_category_embedding_model_remote_code_allowlist():
    assert category_embedding_model_requires_remote_code("Alibaba-NLP/gte-large-en-v1.5") is True
    assert category_embedding_model_requires_remote_code("jinaai/jina-embeddings-v3") is True
    assert category_embedding_model_requires_remote_code("google/embeddinggemma-300m") is False


def test_category_embedding_device_policy_disables_mps_for_remote_code_models(monkeypatch):
    monkeypatch.setattr("video_service.core.embedding_models.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "video_service.core.embedding_models.torch.backends.mps.is_available",
        lambda: True,
    )

    assert (
        resolve_category_embedding_device(
            "Alibaba-NLP/gte-large-en-v1.5",
            preferred_device="mps",
        )
        == "cpu"
    )
    assert (
        resolve_category_embedding_device(
            "jinaai/jina-embeddings-v3",
            preferred_device="mps",
        )
        == "cpu"
    )


def test_category_embedding_device_policy_uses_cuda_when_allowed(monkeypatch):
    monkeypatch.setattr("video_service.core.embedding_models.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "video_service.core.embedding_models.torch.backends.mps.is_available",
        lambda: False,
    )

    assert (
        resolve_category_embedding_device(
            "Alibaba-NLP/gte-large-en-v1.5",
            preferred_device="cuda",
        )
        == "cuda"
    )
    assert (
        resolve_category_embedding_device(
            "google/embeddinggemma-300m",
            preferred_device="cuda",
        )
        == "cuda"
    )
