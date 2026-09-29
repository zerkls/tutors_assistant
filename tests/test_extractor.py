"""Тесты извлечения полей: даты, длительность сна, ID, автор, зона."""

import pytest

from tutor_assistant.extractor import (
    extract,
    extract_author,
    extract_child_id,
    extract_date,
    extract_sleep_hours,
    extract_zone,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("01.03.2025 наблюдение", "2025-03-01"),
        ("07.03.25 ch-0421", "2025-03-07"),
        ("Отчёт от 1 марта 2025 г.", "2025-03-01"),
        ("Рекомендация от 12 марта 2025 г.", "2025-03-12"),
        ("03/01/25, мама ребёнка", "2025-03-01"),
        ("03/01/2025", "2025-03-01"),
        ("запись 2025-03-01", "2025-03-01"),
        ("1 МАРТА 2025", "2025-03-01"),
    ],
)
def test_extract_date_formats(text: str, expected: str) -> None:
    """Все основные форматы дат приводятся к ISO."""
    assert extract_date(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Ребёнок в хорошем настроении",
        "31.02.2025 несуществующая дата",
        "спал 6,5 часов",
        "занятие №12",
    ],
)
def test_extract_date_returns_none(text: str) -> None:
    """Без корректной даты возвращается None, числа сна и номера не путаются с датой."""
    assert extract_date(text) is None


def test_extract_date_takes_first_in_text() -> None:
    """Если дат несколько, берётся первая по тексту."""
    assert extract_date("05.03.2025, перенос на 10.03.2025") == "2025-03-05"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Сон 6 ч 30 мин", 6.5),
        ("спал 6,5 часов", 6.5),
        ("спал 6.5 часа", 6.5),
        ("итого 5:45 сна", 5.75),
        ("сон~390 минут", 6.5),
        ("спал 8 ч.", 8.0),
        ("заснул только к полуночи, спал 5 часов", 5.0),
        ("лёг в 22:30, сон 8 часов", 8.0),
    ],
)
def test_extract_sleep_hours_formats(text: str, expected: float) -> None:
    """Разные форматы длительности переводятся в часы."""
    assert extract_sleep_hours(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "плакал 20 минут без видимой причины",
        "поездка в центр на автобусе заняла 40 минут",
        "сон нормальный",
        "лёг спать в 22:30",
        "спал 30 часов",
    ],
)
def test_extract_sleep_hours_ignores_non_sleep(text: str) -> None:
    """Длительности без контекста сна, время суток и нереальные значения не считаются сном."""
    assert extract_sleep_hours(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ребёнок CH-0421", "CH-0421"),
        ("ch-0421 тьют", "CH-0421"),
        ("CH0107", "CH-0107"),
        ("без ID", None),
    ],
)
def test_extract_child_id(text: str, expected: str | None) -> None:
    """ID приводится к единому виду CH-XXXX."""
    assert extract_child_id(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("тьютор Иванова А. С. Сон", "Иванова А. С."),
        ("тьют.Иванова сон", "Иванова"),
        ("Специалист Петров И.И.", "Петров И. И."),
        ("03/01/25, мама ребёнка CH-0342", "мама"),
        ("Ребёнок играл", None),
        ("позвонили маме после занятия", None),
    ],
)
def test_extract_author(text: str, expected: str | None) -> None:
    """Автор - специалист с фамилией или родитель в роли автора записи."""
    assert extract_author(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ребёнок CH-0107, зона: моторика.", "моторика"),
        ("Зона - сон", "сон"),
        ("На занятии по коммуникации отвечал односложно", None),
        ("сенсор.перегрузка в столовой", None),
        ("зона: математика", None),
    ],
)
def test_extract_zone_only_explicit(text: str, expected: str | None) -> None:
    """Зона извлекается только при явном указании и только из списка отслеживаемых."""
    assert extract_zone(text) == expected


EXPECTED_RECORDS = {
    "R1": {
        "date": "2025-03-05",
        "child_id": "CH-0421",
        "author": "Иванова А. С.",
        "sleep_hours": 6.5,
        "zone": None,
    },
    "R2": {
        "date": "2025-03-01",
        "child_id": "CH-0107",
        "author": "Петров И. И.",
        "sleep_hours": 6.5,
        "zone": "моторика",
    },
    "R3": {
        "date": "2025-03-01",
        "child_id": "CH-0342",
        "author": "мама",
        "sleep_hours": 5.75,
        "zone": None,
    },
    "R4": {
        "date": "2025-03-07",
        "child_id": "CH-0421",
        "author": "Иванова",
        "sleep_hours": 6.5,
        "zone": None,
    },
    "R5": {
        "date": "2025-03-12",
        "child_id": "CH-0107",
        "author": "Петров И. И.",
        "sleep_hours": None,
        "zone": None,
    },
    "R6": {"date": None, "child_id": None, "author": None, "sleep_hours": None, "zone": None},
}


@pytest.mark.parametrize("record_id", sorted(EXPECTED_RECORDS))
def test_extract_on_dataset(records: dict[str, str], record_id: str) -> None:
    """Функция extract на записях R1-R6 совпадает с ручной разметкой."""
    assert extract(records[record_id]) == EXPECTED_RECORDS[record_id]
