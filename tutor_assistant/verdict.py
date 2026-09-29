"""Структура результата проверки зоны."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ZoneVerdict:
    """Полный результат проверки наблюдения на отнесение к зонам.

    Attributes:
        relevant: относится ли наблюдение хотя бы к одной отслеживаемой зоне.
        confidence: уверенность от 0.0 до 1.0.
        reason: понятное специалисту объяснение вывода.
        zones: найденные зоны, основная - первая.
        source: кто принял решение: 'llm' или 'fallback'.
    """

    relevant: bool
    confidence: float
    reason: str
    zones: tuple[str, ...] = ()
    source: str = "fallback"

    def as_tuple(self) -> tuple[bool, float, str]:
        """Возвращает результат в формате check_zone: (relevant, confidence, reason)."""
        return self.relevant, self.confidence, self.reason


def format_zones(zones: tuple[str, ...]) -> str:
    """Формирует фразу "Относится к зоне 'питание'" или "Относится к зонам 'питание', 'сон'"."""
    quoted = ", ".join(f"'{zone}'" for zone in zones)
    noun = "зоне" if len(zones) == 1 else "зонам"
    return f"Относится к {noun} {quoted}"
