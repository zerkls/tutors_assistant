"""Константы и параметры модуля обработки записей тьютора."""

from typing import Final

TRACKED_ZONES: Final[tuple[str, ...]] = (
    "эмоции",
    "коммуникация",
    "поведение",
    "питание",
    "сенсорика",
    "сон",
    "моторика",
    "самообслуживание",
)

RECORD_TYPES: Final[tuple[str, ...]] = (
    "observation",
    "lesson_report",
    "parent_note",
    "recommendation",
)
UNKNOWN_TYPE: Final[str] = "unknown"

CENTURY_BASE: Final[int] = 2000
SHORT_YEAR_LENGTH: Final[int] = 2
MINUTES_PER_HOUR: Final[int] = 60
MAX_SLEEP_HOURS: Final[float] = 24.0
SLEEP_HOURS_PRECISION: Final[int] = 2
SLEEP_CONTEXT_WINDOW: Final[int] = 25

CLASSIFIER_SMOOTHING: Final[float] = 0.5
UNKNOWN_MARGIN: Final[float] = 0.175
CONFIDENCE_PRECISION: Final[int] = 2

FALLBACK_BASE_CONFIDENCE: Final[float] = 0.5
FALLBACK_CONFIDENCE_STEP: Final[float] = 0.1
FALLBACK_MAX_CONFIDENCE: Final[float] = 0.9
FALLBACK_NON_ZONE_CONFIDENCE: Final[float] = 0.8
FALLBACK_NO_EVIDENCE_CONFIDENCE: Final[float] = 0.55
SECONDARY_ZONE_RATIO: Final[float] = 0.5

REVIEW_CONFIDENCE_THRESHOLD: Final[float] = 0.6
DISAGREEMENT_CONFIDENCE: Final[float] = 0.5

GIGACHAT_CREDENTIALS_ENV: Final[str] = "GIGACHAT_CREDENTIALS"
GIGACHAT_DEFAULT_MODEL: Final[str] = "GigaChat-2"
LLM_TEMPERATURE: Final[float] = 0.1
LLM_MAX_TOKENS: Final[int] = 300
LLM_TIMEOUT_SECONDS: Final[float] = 30.0
