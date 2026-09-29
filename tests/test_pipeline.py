"""Тесты сквозного пайплайна."""

import json
from typing import Any

from tutor_assistant.pipeline import run_pipeline


def test_pipeline_report_structure(dataset: dict[str, Any]) -> None:
    """Отчёт содержит все записи и наблюдения, сериализуется в JSON и считает долю уточнений."""
    report = run_pipeline(dataset, use_llm=False)
    assert report["summary"]["records_total"] == len(dataset["records"])
    assert report["summary"]["observations_total"] == len(dataset["observations"])
    assert 0.0 <= report["summary"]["records_needs_review_share"] <= 1.0
    json.dumps(report, ensure_ascii=False)


def test_record_without_metadata_needs_review(dataset: dict[str, Any]) -> None:
    """Запись R6 без даты и ID ребёнка помечается как требующая уточнения."""
    report = run_pipeline(dataset, use_llm=False)
    r6 = next(record for record in report["records"] if record["id"] == "R6")
    assert r6["needs_review"] is True
    assert "не найдено поле child_id" in r6["review_reasons"]
