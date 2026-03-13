from pathlib import Path

import pytest

from video_service.core.category_mapping import (
    build_product_cue_query_text,
    load_category_mapping,
    select_mapping_input_text,
)
from video_service.core.categories import _prepare_query_text_for_embedding

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


def test_select_mapping_input_text_prefers_evidence_for_generic_freeform_category():
    assert (
        select_mapping_input_text(
            raw_category="Technology / Internet Services",
            predicted_brand="Google",
            ocr_summary="Google Pixel 9",
            exact_taxonomy_match=False,
        )
        == "Google\nGoogle Pixel 9"
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


def test_prepare_query_text_for_embedding_prefixes_only_bge_large_en_v15():
    raw_query = "Over-the-Counter Medication"

    assert _prepare_query_text_for_embedding(
        raw_query,
        "BAAI/bge-large-en-v1.5",
    ) == "Represent this sentence for searching relevant passages: Over-the-Counter Medication"
    assert _prepare_query_text_for_embedding(
        raw_query,
        "sentence-transformers/all-mpnet-base-v2",
    ) == raw_query
