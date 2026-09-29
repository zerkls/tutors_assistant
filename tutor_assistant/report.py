"""Отрисовка результатов пайплайна в markdown-таблицы."""

from typing import Any

from tutor_assistant.extractor import ExtractedFields

EMPTY_CELL = "-"
FIELD_NAMES: tuple[str, ...] = tuple(ExtractedFields.__annotations__)


def _cell(value: Any) -> str:
    """Готовит значение для ячейки: None -> '-', экранирует вертикальную черту."""
    if value is None:
        return EMPTY_CELL
    return str(value).replace("|", "\\|")


def _table(header: list[str], rows: list[list[Any]]) -> str:
    """Собирает markdown-таблицу."""
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def extraction_table(report: dict[str, Any]) -> str:
    """Таблица полей, извлечённых из R1-R6."""
    rows = [[r["id"], *(r["fields"][name] for name in FIELD_NAMES)] for r in report["records"]]
    return _table(["ID", *FIELD_NAMES], rows)


def classification_table(report: dict[str, Any]) -> str:
    """Таблица типов записей R1-R6."""
    rows = [[r["id"], r["type"]["label"], r["type"]["confidence"]] for r in report["records"]]
    return _table(["ID", "Тип", "Уверенность"], rows)


def zones_table(report: dict[str, Any]) -> str:
    """Таблица проверки 15 наблюдений на отнесение к зонам."""
    rows = [
        [index, o["text"], o["relevant"], o["confidence"], o["reason"]]
        for index, o in enumerate(report["observations"], start=1)
    ]
    return _table(["№", "Наблюдение", "Относится", "Уверенность", "Объяснение"], rows)


def review_table(report: dict[str, Any]) -> str:
    """Таблица записей R1-R6 с пометкой 'требуется уточнение'."""
    rows = [
        [r["id"], "да" if r["needs_review"] else "нет", "; ".join(r["review_reasons"]) or None]
        for r in report["records"]
    ]
    return _table(["ID", "Требуется уточнение", "Причины"], rows)


def sweep_table(results: list[dict[str, float]]) -> str:
    """Таблица перебора порога unknown."""
    rows = [[r["margin"], r["accuracy"], r["unknown_share"]] for r in results]
    return _table(["Порог", "Точность", "Доля unknown"], rows)


def render_markdown(report: dict[str, Any]) -> str:
    """Собирает все таблицы отчёта в один markdown-документ."""
    summary = report["summary"]
    sections = [
        "## Часть 1. Извлечение полей\n\n" + extraction_table(report),
        "## Часть 2. Классификация\n\n" + classification_table(report),
        "## Часть 3. Проверка зон\n\n" + zones_table(report),
        "## Требуется уточнение\n\n"
        + review_table(report)
        + f"\n\nДоля записей R1-R6: {summary['records_needs_review_share']}, "
        + f"доля наблюдений: {summary['observations_needs_review_share']}. "
        + f"Источник решений по зонам: {', '.join(summary['zone_sources'])}.",
    ]
    return "\n\n".join(sections) + "\n"
