"""Тесты проверки зон: запасной режим, разбор ответа LLM и переключение режимов."""

import pytest

from tutor_assistant.llm import parse_llm_response
from tutor_assistant.zones import analyze_zone, check_zone

EXPECTED_RELEVANCE = {
    "отказ от еды третий день подряд": ("питание",),
    "ударил себя по голове при смене маршрута": ("поведение",),
    "не отвечал на обращённую речь весь день": ("коммуникация",),
    "закрывал уши в столовой из-за шума": ("сенсорика",),
    "заснул только к полуночи, спал 5 часов": ("сон",),
    "впервые сам застегнул куртку": ("самообслуживание",),
    "не удерживает карандаш, роняет мелкие предметы": ("моторика",),
    "плакал 20 минут без видимой причины": ("эмоции",),
    "поездка в центр на автобусе заняла 40 минут": (),
    "родители перенесли занятие на четверг": (),
    "в кабинете меняли лампы, занятие прошло в другой комнате": (),
    "оформлена справка для поликлиники": (),
    "ел только жёлтую еду, остальное отодвигал": ("питание",),
    "отказался идти на занятие, потому что шёл дождь": ("поведение",),
    "был весёлым на празднике, много бегал и шумел": ("эмоции",),
}


@pytest.mark.parametrize(("note", "expected_zones"), EXPECTED_RELEVANCE.items())
def test_fallback_on_observations(note: str, expected_zones: tuple[str, ...]) -> None:
    """Запасной режим верно определяет отнесение и основную зону для 15 наблюдений."""
    verdict = analyze_zone(note, use_llm=False)
    assert verdict.relevant is bool(expected_zones)
    assert verdict.zones[:1] == expected_zones[:1]
    assert verdict.source == "fallback"


def test_check_zone_returns_tuple_with_reason() -> None:
    """check_zone возвращает (bool, float, str) с понятным объяснением."""
    relevant, confidence, reason = check_zone("отказ от еды третий день подряд", use_llm=False)
    assert relevant is True
    assert 0.0 <= confidence <= 1.0
    assert "'питание'" in reason


def test_canteen_is_not_nutrition() -> None:
    """Слово 'столовая' само по себе не делает наблюдение пищевым."""
    verdict = analyze_zone("закрывал уши в столовой из-за шума", use_llm=False)
    assert "питание" not in verdict.zones


def test_unclear_note_needs_clarification() -> None:
    """Без признаков зон и организационных маркеров уверенность низкая."""
    relevant, confidence, reason = check_zone("всё как обычно", use_llm=False)
    assert relevant is False
    assert confidence < 0.6
    assert "уточнение" in reason


class FakeModel:
    """Заглушка LLM, возвращающая заранее заданный ответ."""

    def __init__(self, answer: str) -> None:
        """Сохраняет ответ, который вернёт модель."""
        self.answer = answer
        self.calls = 0

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Возвращает заданный ответ и считает вызовы."""
        self.calls += 1
        return self.answer


def test_llm_answer_is_used() -> None:
    """Корректный ответ LLM попадает в результат как есть."""
    model = FakeModel(
        '```json\n{"relevant": true, "zones": ["Питание"], "confidence": 0.93, '
        '"reason": "Отказ от еды относится к зоне \'питание\'"}\n```'
    )
    verdict = analyze_zone("отказ от еды третий день подряд", model=model)
    assert verdict.as_tuple() == (True, 0.93, "Отказ от еды относится к зоне 'питание'")
    assert verdict.zones == ("питание",)
    assert verdict.source == "llm"


@pytest.mark.parametrize(
    "answer",
    [
        "не могу ответить",
        '{"relevant": "да", "zones": [], "confidence": 0.9, "reason": "..."}',
        '{"relevant": true, "zones": ["математика"], "confidence": 0.9, "reason": "..."}',
    ],
)
def test_invalid_llm_answer_falls_back(answer: str) -> None:
    """Некорректный ответ LLM не ломает пайплайн: включается запасной режим."""
    model = FakeModel(answer)
    verdict = analyze_zone("отказ от еды третий день подряд", model=model)
    assert model.calls == 1
    assert verdict.source == "fallback"
    assert verdict.zones == ("питание",)


def test_llm_error_falls_back() -> None:
    """Ошибка сети или авторизации тоже приводит к запасному режиму."""

    class BrokenModel:
        def complete(self, system_prompt: str, user_message: str) -> str:
            raise ConnectionError("нет сети")

    verdict = analyze_zone("плакал 20 минут без видимой причины", model=BrokenModel())
    assert verdict.source == "fallback"
    assert verdict.zones == ("эмоции",)


def test_no_credentials_uses_fallback() -> None:
    """Без ключа GIGACHAT_CREDENTIALS даже режим LLM работает на ключевых словах."""
    assert analyze_zone("плакал без причины", use_llm=True).source == "fallback"


def test_parse_clamps_confidence() -> None:
    """Уверенность вне диапазона обрезается до [0, 1]."""
    verdict = parse_llm_response(
        '{"relevant": false, "zones": [], "confidence": 1.7, "reason": "Организационное событие"}'
    )
    assert verdict.confidence == 1.0
    assert verdict.zones == ()


def test_network_error_disables_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """После сетевой ошибки клиент по умолчанию больше не вызывается до конца запуска."""
    from tutor_assistant import llm

    class OfflineModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system_prompt: str, user_message: str) -> str:
            self.calls += 1
            raise ConnectionError("нет сети")

    offline = OfflineModel()
    monkeypatch.setattr(llm, "_create_default_model", lambda: offline)
    first = analyze_zone("плакал без причины", use_llm=True)
    second = analyze_zone("отказ от еды", use_llm=True)
    assert first.source == second.source == "fallback"
    assert offline.calls == 1


def test_llm_miss_is_sent_to_review() -> None:
    """Если LLM отрицает зону, а словарь находит признаки, запись уходит на уточнение."""
    model = FakeModel(
        '{"relevant": false, "zones": [], "confidence": 1.0, "reason": "Организационное событие."}'
    )
    verdict = analyze_zone("ударил себя по голове при смене маршрута", model=model)
    assert verdict.relevant is False
    assert verdict.confidence < 0.6
    assert "Требуется уточнение" in verdict.reason
    assert "'поведение'" in verdict.reason


def test_placeholder_reason_is_replaced() -> None:
    """Заглушка вместо объяснения заменяется понятной фразой."""
    verdict = parse_llm_response(
        '{"relevant": true, "zones": ["сон"], "confidence": 0.9, "reason": "..."}'
    )
    assert verdict.reason == "Относится к зоне 'сон'"
