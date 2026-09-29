"""Общие фикстуры тестов."""

import json
from pathlib import Path
from typing import Any

import pytest

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset.json"


@pytest.fixture(scope="session")
def dataset() -> dict[str, Any]:
    """Тестовый датасет из приложения 1."""
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def records(dataset: dict[str, Any]) -> dict[str, str]:
    """Записи R1-R6 в виде словаря id -> текст."""
    return {record["id"]: record["text"] for record in dataset["records"]}


@pytest.fixture(autouse=True)
def no_llm_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Тесты не ходят в GigaChat, даже если ключ задан в окружении."""
    from tutor_assistant.llm import reset_default_model

    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    reset_default_model()
