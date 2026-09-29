"""Клиент GigaChat и разбор ответа модели для проверки зон."""

import json
import os
import re
from functools import lru_cache
from typing import Protocol

from tutor_assistant.config import (
    CONFIDENCE_PRECISION,
    GIGACHAT_CREDENTIALS_ENV,
    GIGACHAT_DEFAULT_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    TRACKED_ZONES,
)
from tutor_assistant.verdict import ZoneVerdict, format_zones

GIGACHAT_MODEL_ENV = "GIGACHAT_MODEL"
LLM_SOURCE = "llm"

SYSTEM_PROMPT = f"""Ты помогаешь команде сопровождения детей с ОВЗ разбирать наблюдения тьютора.
Команда отслеживает строго эти зоны развития: {", ".join(TRACKED_ZONES)}.

Определи, описывает ли наблюдение состояние, реакцию, навык или поведение ребёнка,
которые относятся хотя бы к одной из этих зон.

Правила:
- Любое действие, реакция или состояние ребёнка относится к зоне, даже если его вызвало
  внешнее событие (смена маршрута или расписания, шум, погода, праздник). Внешнее событие -
  это причина реакции, а не признак организационной записи.
- Самоповреждение, агрессия, отказы и протесты относятся к зоне 'поведение'.
- Место (столовая, кабинет, улица) не определяет зону: смотри, что делал или чувствовал
  ребёнок. Реакции на звуки, свет, запахи и прикосновения относятся к зоне 'сенсорика'.
- К зонам не относятся только записи, где нет действий и состояния ребёнка:
  транспорт, перенос занятий, документы, ремонт, замена оборудования или сотрудников.
- Обычное, не проблемное поведение тоже относится к зоне.
- Указывай только зоны из списка, основную зону ставь первой.
- confidence: 0.85-0.95 для однозначных случаев, 0.6-0.8 при разных толкованиях,
  1.0 не ставь.
- reason: одно предложение для специалиста - что сделал или чувствовал ребёнок
  и почему это относится к зоне. Без технических терминов.

Отвечай строго одним JSON-объектом без текста вокруг. Примеры:

Наблюдение: кусал рукав, когда в классе включили музыку
{{"relevant": true, "zones": ["поведение", "сенсорика"], "confidence": 0.85,
"reason": "Ребёнок кусал рукав в ответ на громкую музыку - поведенческая реакция на звук."}}

Наблюдение: тьютор ушла на больничный, неделю работает замена
{{"relevant": false, "zones": [], "confidence": 0.9,
"reason": "Описана замена сотрудника, состояние и действия ребёнка не упоминаются."}}"""


class ChatModel(Protocol):
    """Минимальный интерфейс языковой модели: системный промпт и текст -> ответ."""

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Возвращает текст ответа модели."""
        ...


class GigaChatModel:
    """Обёртка над официальным SDK gigachat.

    Ключ передаётся явно, остальные настройки (scope, проверка сертификатов)
    SDK читает из переменных окружения GIGACHAT_*.
    """

    def __init__(self, credentials: str, model: str = GIGACHAT_DEFAULT_MODEL) -> None:
        """Создаёт клиента GigaChat."""
        from gigachat import GigaChat

        self._client = GigaChat(credentials=credentials, model=model, timeout=LLM_TIMEOUT_SECONDS)

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Отправляет запрос в GigaChat и возвращает текст ответа."""
        from gigachat.models import Chat, Messages, MessagesRole

        response = self._client.chat(
            Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=system_prompt),
                    Messages(role=MessagesRole.USER, content=user_message),
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        )
        return response.choices[0].message.content


class LLMUnavailableError(Exception):
    """Модель недоступна: нет сети, ошибка авторизации или сервера."""


_disabled_reasons: list[str] = []


def disable_default_model(reason: str) -> None:
    """Отключает GigaChat до конца запуска, чтобы не ждать таймаут на каждой записи."""
    _disabled_reasons.append(reason)


def reset_default_model() -> None:
    """Сбрасывает кэш клиента и отметку об отключении."""
    _disabled_reasons.clear()
    _create_default_model.cache_clear()


def get_default_model() -> ChatModel | None:
    """Возвращает клиента GigaChat, если задан ключ, установлен SDK и модель не отключена."""
    if _disabled_reasons:
        return None
    return _create_default_model()


@lru_cache(maxsize=1)
def _create_default_model() -> ChatModel | None:
    """Создаёт клиента GigaChat по переменным окружения."""
    credentials = os.getenv(GIGACHAT_CREDENTIALS_ENV)
    if not credentials:
        return None
    try:
        return GigaChatModel(credentials, os.getenv(GIGACHAT_MODEL_ENV, GIGACHAT_DEFAULT_MODEL))
    except ImportError:
        return None


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
MEANINGFUL_TEXT_RE = re.compile(r"[а-яёa-z]{3}", re.IGNORECASE)


def parse_llm_response(content: str) -> ZoneVerdict:
    """Разбирает JSON-ответ модели и проверяет его на соответствие контракту.

    Raises:
        ValueError: если ответ не содержит JSON, поля имеют неверный тип
            или модель отнесла запись к зоне, которой нет в списке.
    """
    match = JSON_OBJECT_RE.search(content)
    if not match:
        raise ValueError("в ответе модели нет JSON")
    payload = json.loads(match.group(0))

    relevant = payload.get("relevant")
    confidence = payload.get("confidence")
    raw_zones = payload.get("zones") or []
    reason = payload.get("reason")
    if not isinstance(relevant, bool):
        raise ValueError("поле relevant должно быть bool")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("поле confidence должно быть числом")
    if not isinstance(raw_zones, list):
        raise ValueError("поле zones должно быть списком")

    zones = tuple(
        zone for zone in (str(item).strip().lower() for item in raw_zones) if zone in TRACKED_ZONES
    )
    if relevant and not zones:
        raise ValueError("модель не назвала ни одной зоны из списка")
    if not relevant:
        zones = ()

    if not isinstance(reason, str) or not MEANINGFUL_TEXT_RE.search(reason):
        reason = format_zones(zones) if zones else "Не относится к отслеживаемым зонам"

    return ZoneVerdict(
        relevant=relevant,
        confidence=round(min(max(float(confidence), 0.0), 1.0), CONFIDENCE_PRECISION),
        reason=reason.strip(),
        zones=zones,
        source=LLM_SOURCE,
    )


def llm_zone_check(note: str, model: ChatModel) -> ZoneVerdict:
    """Проверяет отнесение наблюдения к зонам с помощью языковой модели.

    Raises:
        LLMUnavailableError: если запрос к модели не удался (сеть, авторизация, сервер).
        ValueError: если модель ответила, но ответ не соответствует контракту.
    """
    try:
        content = model.complete(SYSTEM_PROMPT, note)
    except Exception as error:
        raise LLMUnavailableError(f"{type(error).__name__}: {error}") from error
    return parse_llm_response(content)
