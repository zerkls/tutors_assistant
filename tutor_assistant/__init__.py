"""Модуль интеллектуальной обработки записей тьютора."""

from tutor_assistant.classifier import classify
from tutor_assistant.extractor import extract
from tutor_assistant.zones import check_zone

__all__ = ["check_zone", "classify", "extract"]
