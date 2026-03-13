from typing import List, Optional, Any
from pydantic import BaseModel, Field, model_validator
from enum import Enum

class JobMode(str, Enum):
    pipeline = "pipeline"
    agent = "agent"
    benchmark = "benchmark"


def _normalize_ocr_mode_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "detail" in text:
        return "Detailed"
    if "fast" in text:
        return "Fast"
    return "Fast"

class JobSettings(BaseModel):
    categories: str = ""
    provider: str = "Gemini CLI"
    model_name: str = "Gemini CLI Default"
    category_embedding_model: str = "BAAI/bge-large-en-v1.5"
    ocr_engine: str = "EasyOCR"
    ocr_mode: str = "Fast"
    scan_mode: str = "Tail Only"
    express_mode: bool = False
    override: bool = False
    enable_search: bool = True
    enable_web_search: Optional[bool] = None
    enable_agentic_search: Optional[bool] = None
    enable_vision_board: bool = True
    enable_llm_frame: bool = True
    context_size: int = 8192

    @model_validator(mode="before")
    @classmethod
    def _normalize_search_aliases(cls, data):
        if not isinstance(data, dict):
            return data

        enable_search = data.get("enable_search")
        enable_web_search = data.get("enable_web_search")
        enable_agentic_search = data.get("enable_agentic_search")
        enable_vision_board = data.get("enable_vision_board")
        enable_llm_frame = data.get("enable_llm_frame")
        legacy_enable_vision = data.get("enable_vision")

        if enable_search is None:
            if enable_web_search is not None:
                data["enable_search"] = bool(enable_web_search)
            elif enable_agentic_search is not None:
                data["enable_search"] = bool(enable_agentic_search)
        if enable_web_search is None:
            data["enable_web_search"] = bool(data.get("enable_search", True))
        if enable_agentic_search is None:
            data["enable_agentic_search"] = bool(data.get("enable_search", True))

        # Deprecated legacy alias:
        # `enable_vision` drove both behaviors before the split.
        if enable_vision_board is None and legacy_enable_vision is not None:
            data["enable_vision_board"] = bool(legacy_enable_vision)
        if enable_llm_frame is None and legacy_enable_vision is not None:
            data["enable_llm_frame"] = bool(legacy_enable_vision)

        if data.get("enable_vision_board") is None:
            data["enable_vision_board"] = True
        if data.get("enable_llm_frame") is None:
            data["enable_llm_frame"] = True
        data["ocr_mode"] = _normalize_ocr_mode_value(data.get("ocr_mode"))
        return data

class JobSettingsForm(JobSettings):
    mode: JobMode = JobMode.pipeline

class UrlBatchRequest(BaseModel):
    mode: JobMode = JobMode.pipeline
    urls: List[str]
    settings: JobSettings

class FolderRequest(BaseModel):
    mode: JobMode = JobMode.pipeline
    folder_path: str
    settings: JobSettings

class FilePathRequest(BaseModel):
    mode: JobMode = JobMode.pipeline
    file_path: str
    settings: JobSettings

class JobResponse(BaseModel):
    job_id: str
    status: str

class BulkDeleteRequest(BaseModel):
    job_ids: List[str]

class JobStatus(BaseModel):
    job_id: str
    status: str
    stage: Optional[str] = None
    stage_detail: Optional[str] = None
    duration_seconds: Optional[float] = None
    created_at: str
    updated_at: str
    progress: float
    error: Optional[str]
    settings: Optional[JobSettings]
    mode: Optional[JobMode]
    url: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    category_id: Optional[str] = None


class BenchmarkTruthCreateRequest(BaseModel):
    name: str
    video_url: str
    expected_ocr_text: str = ""
    expected_categories: List[str] = Field(default_factory=list)
    expected_brand: str = ""
    expected_category: str = ""
    expected_confidence: Optional[float] = None
    expected_reasoning: str = ""
    metadata: dict = Field(default_factory=dict)


class BenchmarkRunRequest(BaseModel):
    truth_id: str
    categories: str = ""
    providers: Optional[List[str]] = None
    models: Optional[List[str]] = None
    # Explicit (provider, model) pairs — bypasses Cartesian auto-resolve when set
    model_combos: Optional[List[dict]] = None
    # Run each benchmark job in express (fast) mode
    express_mode: bool = False


class BenchmarkSuiteUpdateRequest(BaseModel):
    name: str
    description: str = ""


class BenchmarkTestUpdateRequest(BaseModel):
    source_url: Optional[str] = None
    expected_category: Optional[str] = None
    expected_brand: Optional[str] = None
    expected_confidence: Optional[float] = None
    expected_reasoning: Optional[str] = None
    expected_ocr_text: Optional[str] = None
    expected_categories: Optional[List[str]] = None
