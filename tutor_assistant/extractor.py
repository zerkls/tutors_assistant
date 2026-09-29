"""Часть 1: извлечение структурированных полей из свободного текста записи."""

import re
from datetime import date
from typing import TypedDict

from tutor_assistant.config import (
    CENTURY_BASE,
    MAX_SLEEP_HOURS,
    MINUTES_PER_HOUR,
    SHORT_YEAR_LENGTH,
    SLEEP_CONTEXT_WINDOW,
    SLEEP_HOURS_PRECISION,
    TRACKED_ZONES,
)


class ExtractedFields(TypedDict):
    """Результат работы extract: отсутствующие поля равны None."""

    date: str | None
    child_id: str | None
    author: str | None
    sleep_hours: float | None
    zone: str | None


MONTHS_GENITIVE: dict[str, int] = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}

DOTTED_DATE_RE = re.compile(r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})(?![\d])")
SLASHED_DATE_RE = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\d/])")
ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
TEXT_DATE_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s+(" + "|".join(MONTHS_GENITIVE) + r")\s+(\d{4})(?!\d)",
    re.IGNORECASE,
)

CHILD_ID_RE = re.compile(r"\b(CH)[-\s]?(\d{3,})\b", re.IGNORECASE)

SURNAME = r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"
INITIALS = r"(?:\s*[А-ЯЁ]\.){1,2}"
SPECIALIST_ROLES = (
    "тьютор",
    "тьют\\.",
    "специалист",
    "нейропсихолог",
    "психолог",
    "логопед",
    "дефектолог",
    "педагог",
    "воспитатель",
)
SPECIALIST_RE = re.compile(
    r"(?i:\b(?:" + "|".join(SPECIALIST_ROLES) + r"))"
    r"\s*(?P<surname>" + SURNAME + r")(?P<initials>" + INITIALS + r")?"
)
PARENT_RE = re.compile(
    r"(?:^|[,.;:]\s*)(?P<parent>мама|папа|мать|отец|бабушка|дедушка|родители|родитель)\b",
    re.IGNORECASE,
)

DURATION_RE = re.compile(
    r"(?P<hm_h>\d{1,2})\s*ч(?:ас(?:а|ов)?)?\.?\s*(?P<hm_m>\d{1,2})\s*мин(?:ут[аы]?)?\.?"
    r"|(?P<clock_h>\d{1,2}):(?P<clock_m>[0-5]\d)"
    r"|(?P<hours>\d+(?:[.,]\d+)?)\s*ч(?:ас(?:а|ов)?)?(?![а-яё])"
    r"|(?P<minutes>\d+)\s*мин(?:ут[аы]?)?(?![а-яё])",
    re.IGNORECASE,
)
SLEEP_KEYWORD_RE = re.compile(r"\bсон|\bсна\b|\bсну\b|спал|засн|уснул", re.IGNORECASE)
TIME_OF_DAY_PREFIX_RE = re.compile(r"(?:\bв|\bк|\bдо|\bс|\bпосле|\bоколо)\s*$", re.IGNORECASE)

ZONE_RE = re.compile(
    r"\bзон[аы]\s*[:\-\u2013\u2014]?\s*[\"'\u00ab]?(?P<zone>[а-яё]+)",
    re.IGNORECASE,
)


def _expand_year(raw_year: str) -> int:
    """Превращает двузначный год в четырёхзначный (25 -> 2025)."""
    year = int(raw_year)
    return year + CENTURY_BASE if len(raw_year) == SHORT_YEAR_LENGTH else year


def _safe_date(year: int, month: int, day: int) -> date | None:
    """Создаёт дату или возвращает None, если такой даты не существует."""
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_date(text: str) -> str | None:
    """Находит первую по положению в тексте корректную дату и возвращает её в ISO.

    Поддерживаемые форматы: 01.03.2025 и 01.03.25 (день.месяц.год),
    03/01/25 и 03/01/2025 (месяц/день/год, американский формат),
    2025-03-01 и 1 марта 2025 г.
    """
    candidates: list[tuple[int, date]] = []

    for match in DOTTED_DATE_RE.finditer(text):
        day, month, year = match.groups()
        parsed = _safe_date(_expand_year(year), int(month), int(day))
        if parsed:
            candidates.append((match.start(), parsed))

    for match in SLASHED_DATE_RE.finditer(text):
        month, day, year = match.groups()
        parsed = _safe_date(_expand_year(year), int(month), int(day))
        if parsed:
            candidates.append((match.start(), parsed))

    for match in ISO_DATE_RE.finditer(text):
        year, month, day = match.groups()
        parsed = _safe_date(int(year), int(month), int(day))
        if parsed:
            candidates.append((match.start(), parsed))

    for match in TEXT_DATE_RE.finditer(text):
        day, month_name, year = match.groups()
        parsed = _safe_date(int(year), MONTHS_GENITIVE[month_name.lower()], int(day))
        if parsed:
            candidates.append((match.start(), parsed))

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1].isoformat()


def extract_child_id(text: str) -> str | None:
    """Находит идентификатор ребёнка и приводит его к виду CH-0421."""
    match = CHILD_ID_RE.search(text)
    if not match:
        return None
    return f"CH-{match.group(2)}"


def _format_initials(raw_initials: str | None) -> str:
    """Приводит инициалы к единому виду: 'А.С.' и 'А. С.' -> 'А. С.'."""
    if not raw_initials:
        return ""
    letters = re.findall(r"[А-ЯЁ]", raw_initials)
    return " " + " ".join(f"{letter}." for letter in letters)


def extract_author(text: str) -> str | None:
    """Находит автора записи: специалиста по фамилии или родителя по роли.

    Для специалиста возвращается фамилия с инициалами ('Иванова А. С.'),
    для родителя - роль в нижнем регистре ('мама').
    Если найдено несколько кандидатов, берётся первый по тексту.
    """
    candidates: list[tuple[int, str]] = []

    specialist = SPECIALIST_RE.search(text)
    if specialist:
        name = specialist.group("surname") + _format_initials(specialist.group("initials"))
        candidates.append((specialist.start(), name))

    parent = PARENT_RE.search(text)
    if parent:
        candidates.append((parent.start("parent"), parent.group("parent").lower()))

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _duration_to_hours(match: re.Match[str]) -> float:
    """Переводит найденную длительность в часы."""
    if match.group("hm_h"):
        return int(match.group("hm_h")) + int(match.group("hm_m")) / MINUTES_PER_HOUR
    if match.group("clock_h"):
        return int(match.group("clock_h")) + int(match.group("clock_m")) / MINUTES_PER_HOUR
    if match.group("hours"):
        return float(match.group("hours").replace(",", "."))
    return int(match.group("minutes")) / MINUTES_PER_HOUR


def _is_time_of_day(text: str, match: re.Match[str]) -> bool:
    """Проверяет, что запись вида ЧЧ:ММ - это время суток ('в 22:30'), а не длительность."""
    return bool(match.group("clock_h")) and bool(
        TIME_OF_DAY_PREFIX_RE.search(text[: match.start()])
    )


def _gap(first: re.Match[str], second: re.Match[str]) -> int:
    """Считает расстояние в символах между двумя совпадениями."""
    return max(0, first.start() - second.end(), second.start() - first.end())


def extract_sleep_hours(text: str) -> float | None:
    """Находит длительность сна в часах.

    Длительность учитывается, только если рядом (в пределах SLEEP_CONTEXT_WINDOW
    символов) есть слово про сон. Так '40 минут в автобусе' не считается сном.
    При нескольких кандидатах берётся ближайший к слову про сон.
    """
    keywords = list(SLEEP_KEYWORD_RE.finditer(text))
    if not keywords:
        return None

    best: tuple[int, float] | None = None
    for duration in DURATION_RE.finditer(text):
        if _is_time_of_day(text, duration):
            continue
        hours = _duration_to_hours(duration)
        if not 0 < hours <= MAX_SLEEP_HOURS:
            continue
        distance = min(_gap(duration, keyword) for keyword in keywords)
        if distance > SLEEP_CONTEXT_WINDOW:
            continue
        if best is None or distance < best[0]:
            best = (distance, hours)

    if best is None:
        return None
    return round(best[1], SLEEP_HOURS_PRECISION)


def extract_zone(text: str) -> str | None:
    """Возвращает зону, только если она явно указана в тексте ('зона: моторика').

    Зона должна входить в список отслеживаемых, иначе возвращается None.
    Смысловое определение зоны выполняет check_zone (часть 3).
    """
    for match in ZONE_RE.finditer(text):
        zone = match.group("zone").lower()
        if zone in TRACKED_ZONES:
            return zone
    return None


def extract(text: str) -> ExtractedFields:
    """Извлекает из записи дату, ID ребёнка, автора, длительность сна и явно указанную зону.

    Args:
        text: исходный текст записи в свободной форме.

    Returns:
        Словарь с ключами date, child_id, author, sleep_hours, zone.
        Поля, которые не удалось найти, равны None.
    """
    return ExtractedFields(
        date=extract_date(text),
        child_id=extract_child_id(text),
        author=extract_author(text),
        sleep_hours=extract_sleep_hours(text),
        zone=extract_zone(text),
    )
