"""Тесты классификации типа записи."""

import pytest

from tutor_assistant.classifier import classify, classify_with_margin, type_probabilities

EXPECTED_TYPES = {
    "R1": "observation",
    "R2": "lesson_report",
    "R3": "parent_note",
    "R4": "observation",
    "R5": "recommendation",
    "R6": "observation",
}


@pytest.mark.parametrize("record_id", sorted(EXPECTED_TYPES))
def test_classify_dataset(records: dict[str, str], record_id: str) -> None:
    """Типы записей R1-R6 совпадают с ручной разметкой."""
    record_type, confidence = classify(records[record_id])
    assert record_type == EXPECTED_TYPES[record_id]
    assert 0.0 < confidence <= 1.0


@pytest.mark.parametrize("text", ["", "Встреча перенесена на пятницу."])
def test_classify_without_evidence_is_unknown(text: str) -> None:
    """Без признаков возвращается unknown с нулевой уверенностью."""
    assert classify(text) == ("unknown", 0.0)


def test_classify_ambiguous_is_unknown() -> None:
    """При близких скорах двух типов возвращается unknown."""
    assert (
        classify("Мама рекомендует давать ребёнку больше времени на переодевание.")[0] == "unknown"
    )


def test_margin_controls_unknown(records: dict[str, str]) -> None:
    """Чем выше порог, тем чаще unknown: при пороге 1.0 любая запись становится unknown."""
    assert classify_with_margin(records["R1"], 0.0)[0] == "observation"
    assert classify_with_margin(records["R1"], 1.0)[0] == "unknown"


def test_probabilities_sum_to_one() -> None:
    """Сглаженные вероятности образуют распределение."""
    probabilities = type_probabilities(
        {"observation": 2.0, "lesson_report": 0.0, "parent_note": 1.0}
    )
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(value > 0 for value in probabilities.values())
