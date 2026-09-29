"""Часть 2: классификация типа записи на взвешенных эвристиках."""

import re
from dataclasses import dataclass

from tutor_assistant.config import (
    CLASSIFIER_SMOOTHING,
    CONFIDENCE_PRECISION,
    RECORD_TYPES,
    UNKNOWN_MARGIN,
    UNKNOWN_TYPE,
)
from tutor_assistant.text_utils import normalize


@dataclass(frozen=True)
class Feature:
    """Признак класса: регулярное выражение и вес, который он добавляет к скору."""

    pattern: re.Pattern[str]
    weight: float


def _feature(pattern: str, weight: float) -> Feature:
    """Создаёт признак из строки регулярного выражения."""
    return Feature(re.compile(pattern), weight)


STRONG = 2.5
MEDIUM = 1.5
WEAK = 1.0
HINT = 0.5

FEATURES: dict[str, tuple[Feature, ...]] = {
    "observation": (
        _feature(r"\bтьютор|\bтьют\.", STRONG),
        _feature(r"наблюдени", STRONG),
        _feature(r"\bсон|\bсна\b|спал|засн|просыпал", WEAK),
        _feature(
            r"настроени|\bнастр|вял|плакал|кричал|агресс|перегрузк|играл|"
            r"отвечал|отказал|истерик|спокоен|спокойн|возбужд",
            WEAK,
        ),
    ),
    "lesson_report": (
        _feature(r"отчет", STRONG),
        _feature(r"№\s*\d+", MEDIUM),
        _feature(r"\bзона\s*:", WEAK),
        _feature(r"специалист|логопед|дефектолог|психолог", WEAK),
        _feature(r"упражнени|задани|выполнил|отработ", WEAK),
        _feature(r"заняти", HINT),
    ),
    "parent_note": (
        _feature(r"\bмам[аы]\b|\bпап[аы]\b|\bмать\b|\bотец\b|родител|бабушк|дедушк", STRONG),
        _feature(r"\bдома\b|ночью|вечером|выходны", HINT),
        _feature(r"завтрак|ужин", HINT),
    ),
    "recommendation": (
        _feature(r"рекоменд", STRONG),
        _feature(
            r"\bввести\b|\bследует\b|необходимо|\bнужно\b|\bстоит\b|предлагаю|"
            r"\bиспользовать\b|\bдобавить\b|\bисключить\b",
            MEDIUM,
        ),
        _feature(r"специалист", HINT),
    ),
}


def score_types(text: str) -> dict[str, float]:
    """Считает сырой скор каждого типа: сумму весов сработавших признаков.

    Каждый признак учитывается не больше одного раза, чтобы длинный текст
    с повторами не перевешивал короткий.
    """
    normalized = normalize(text)
    return {
        record_type: sum(f.weight for f in FEATURES[record_type] if f.pattern.search(normalized))
        for record_type in RECORD_TYPES
    }


def type_probabilities(scores: dict[str, float]) -> dict[str, float]:
    """Переводит скоры в распределение со сглаживанием Лапласа.

    Сглаживание не даёт одному слабому признаку дать уверенность 1.0:
    чем меньше свидетельств, тем ближе распределение к равномерному.
    """
    total = sum(scores.values()) + CLASSIFIER_SMOOTHING * len(scores)
    return {name: (score + CLASSIFIER_SMOOTHING) / total for name, score in scores.items()}


def classify_with_margin(text: str, margin: float = UNKNOWN_MARGIN) -> tuple[str, float]:
    """Классифицирует запись с заданным порогом разрыва между первым и вторым местом.

    Args:
        text: исходный текст записи.
        margin: минимальный разрыв вероятностей первого и второго типа.

    Returns:
        Пара (тип, уверенность). Если признаков нет или разрыв меньше margin,
        тип равен 'unknown', а уверенность - вероятности лучшего кандидата.
    """
    scores = score_types(text)
    if not any(scores.values()):
        return UNKNOWN_TYPE, 0.0

    ranked = sorted(type_probabilities(scores).items(), key=lambda item: item[1], reverse=True)
    (best_type, best_prob), (_, second_prob) = ranked[0], ranked[1]
    confidence = round(best_prob, CONFIDENCE_PRECISION)
    if best_prob - second_prob < margin:
        return UNKNOWN_TYPE, confidence
    return best_type, confidence


def classify(text: str) -> tuple[str, float]:
    """Определяет тип записи: observation, lesson_report, parent_note, recommendation или unknown.

    Args:
        text: исходный текст записи.

    Returns:
        Пара (тип, уверенность от 0.0 до 1.0).
    """
    return classify_with_margin(text, UNKNOWN_MARGIN)
