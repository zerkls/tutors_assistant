"""Сквозной пайплайн: классификация -> извлечение полей -> проверка зон -> JSON-отчёт."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tutor_assistant.classifier import classify, classify_with_margin
from tutor_assistant.config import CONFIDENCE_PRECISION, REVIEW_CONFIDENCE_THRESHOLD, UNKNOWN_TYPE
from tutor_assistant.extractor import extract
from tutor_assistant.zones import analyze_zone

REQUIRED_FIELDS: tuple[str, ...] = ("date", "child_id")
SWEEP_STEP = 0.025
SWEEP_MAX = 0.4


def load_dataset(path: Path) -> dict[str, Any]:
    """Читает dataset.json с ключами records (R1-R6) и observations (15 наблюдений)."""
    return json.loads(path.read_text(encoding="utf-8"))


def process_record(record_id: str, text: str, use_llm: bool | None = None) -> dict[str, Any]:
    """Прогоняет одну запись через все три шага и решает, нужна ли ручная проверка."""
    record_type, type_confidence = classify(text)
    fields = extract(text)
    verdict = analyze_zone(text, use_llm)

    review_reasons: list[str] = []
    if record_type == UNKNOWN_TYPE:
        review_reasons.append("не удалось уверенно определить тип записи")
    review_reasons.extend(
        f"не найдено поле {name}" for name in REQUIRED_FIELDS if fields[name] is None
    )
    if verdict.confidence < REVIEW_CONFIDENCE_THRESHOLD:
        review_reasons.append("низкая уверенность в отнесении к зонам")

    return {
        "id": record_id,
        "text": text,
        "type": {"label": record_type, "confidence": type_confidence},
        "fields": dict(fields),
        "zone_check": asdict(verdict),
        "needs_review": bool(review_reasons),
        "review_reasons": review_reasons,
    }


def process_observation(note: str, use_llm: bool | None = None) -> dict[str, Any]:
    """Проверяет одно короткое наблюдение на отнесение к зонам."""
    verdict = analyze_zone(note, use_llm)
    return {
        "text": note,
        **asdict(verdict),
        "needs_review": verdict.confidence < REVIEW_CONFIDENCE_THRESHOLD,
    }


def _share(items: list[dict[str, Any]]) -> float:
    """Доля элементов с пометкой 'требуется уточнение'."""
    if not items:
        return 0.0
    return round(sum(item["needs_review"] for item in items) / len(items), CONFIDENCE_PRECISION)


def run_pipeline(dataset: dict[str, Any], use_llm: bool | None = None) -> dict[str, Any]:
    """Обрабатывает весь датасет и собирает итоговый отчёт со сводкой для руководителя."""
    records = [process_record(r["id"], r["text"], use_llm) for r in dataset.get("records", [])]
    observations = [process_observation(note, use_llm) for note in dataset.get("observations", [])]

    type_distribution: dict[str, int] = {}
    for record in records:
        label = record["type"]["label"]
        type_distribution[label] = type_distribution.get(label, 0) + 1

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "records_total": len(records),
            "records_needs_review_share": _share(records),
            "observations_total": len(observations),
            "observations_needs_review_share": _share(observations),
            "type_distribution": type_distribution,
            "zone_sources": sorted(
                {r["zone_check"]["source"] for r in records} | {o["source"] for o in observations}
            ),
        },
        "records": records,
        "observations": observations,
    }


def sweep_unknown_margin(samples: list[dict[str, str]]) -> list[dict[str, float]]:
    """Перебирает пороги unknown и считает точность и долю unknown на размеченной выборке."""
    results: list[dict[str, float]] = []
    steps = round(SWEEP_MAX / SWEEP_STEP)
    for step in range(steps + 1):
        margin = round(step * SWEEP_STEP, 3)
        predictions = [classify_with_margin(sample["text"], margin)[0] for sample in samples]
        correct = sum(
            pred == sample["label"] for pred, sample in zip(predictions, samples, strict=True)
        )
        unknown = sum(pred == UNKNOWN_TYPE for pred in predictions)
        results.append(
            {
                "margin": margin,
                "accuracy": round(correct / len(samples), CONFIDENCE_PRECISION),
                "unknown_share": round(unknown / len(samples), CONFIDENCE_PRECISION),
            }
        )
    return results
