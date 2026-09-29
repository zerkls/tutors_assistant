"""Часть 3: проверка отнесения наблюдения к отслеживаемым зонам развития."""

import logging
import re
from dataclasses import replace

from tutor_assistant.config import (
    CONFIDENCE_PRECISION,
    DISAGREEMENT_CONFIDENCE,
    FALLBACK_BASE_CONFIDENCE,
    FALLBACK_CONFIDENCE_STEP,
    FALLBACK_MAX_CONFIDENCE,
    FALLBACK_NO_EVIDENCE_CONFIDENCE,
    FALLBACK_NON_ZONE_CONFIDENCE,
    SECONDARY_ZONE_RATIO,
)
from tutor_assistant.llm import (
    ChatModel,
    LLMUnavailableError,
    disable_default_model,
    get_default_model,
    llm_zone_check,
)
from tutor_assistant.text_utils import normalize
from tutor_assistant.verdict import ZoneVerdict, format_zones

logger = logging.getLogger(__name__)

FALLBACK_SOURCE = "fallback"
LOG_ERROR_MAX_LENGTH = 150

ZONE_LEXICON: dict[str, tuple[str, ...]] = {
    "питание": (
        r"\bед(?:а|ы|у|е|ой)\b",
        r"\bел[аи]?\b|\bест\b|съел|\bпоел",
        r"завтрак|\bобед|ужин|перекус",
        r"пищ|аппетит|жева|глота|кормл|\bпить\b|\bпил\b",
    ),
    "сон": (
        r"\bсон\b|\bсна\b|\bсну\b",
        r"спал|засн|засып|\bусн",
        r"просып|бессон|дрем",
        r"полуноч|\bночь|ночью",
    ),
    "сенсорика": (
        r"сенсор|перегрузк",
        r"\bшум(?:а|у|ом|е)?\b|шумн|громк",
        r"закрыва\w*\s+уш|\bуш[иа]\b",
        r"ярк\w*\s+свет|запах|прикоснов|тактил|текстур",
        r"избирательн|только\s+(?:желт|красн|бел|зелен|оранжев)\w*",
    ),
    "эмоции": (
        r"плак|слез|истерик|крич",
        r"злил|злост|раздраж|обид|капризн",
        r"радост|\bрад\b|весел|смеял|улыба",
        r"грус|тревож|страх|боял|испуг",
        r"настроени|\bнастр",
    ),
    "коммуникация": (
        r"\bреч[ьи]\b|говор|сказал|\bслов",
        r"отвеча|обращ|односложн|молчал",
        r"жест|контакт|диалог|общал|общени|коммуникац",
        r"просьб|\bзвал|здорова|карточк",
    ),
    "поведение": (
        r"удар|\bбил[аи]?\b|кусал|толка|щипа|бросал",
        r"агресс|самоповрежд|стереотип|раскачив",
        r"убега|бегал|шумел|не\s+слуша|протест",
        r"отказ\w*(?!\s+от\s+(?:еды|пищи|завтрака|обеда|ужина))",
        r"смен\w*\s+маршрут",
    ),
    "моторика": (
        r"карандаш|ножниц|почерк|рисова|лепи",
        r"\bрон|удерж|\bдерж",
        r"мелк\w*\s+предмет|пуговиц|застег",
        r"ходьб|прыг|равновес|координац|моторик|лестниц|\bмяч",
    ),
    "самообслуживание": (
        r"застег|одева|оделс|раздева|обува|\bобул|куртк|шнур",
        r"умыва|мыл\w*\s+рук|туалет|горшок|чистил\w*\s+зуб",
        r"\bсам\b|\bсама\b|самостоятельн|самообслуж",
        r"ложк|вилк",
    ),
}

NON_ZONE_MARKERS: tuple[str, ...] = (
    r"поездк|автобус|транспорт|такси|маршрутк",
    r"перенесл|перенос|расписани",
    r"справк|поликлиник|документ|оформ|оплат|договор",
    r"кабинет|комнат|ламп|ремонт",
)

COMPILED_LEXICON: dict[str, tuple[re.Pattern[str], ...]] = {
    zone: tuple(re.compile(pattern) for pattern in patterns)
    for zone, patterns in ZONE_LEXICON.items()
}
COMPILED_NON_ZONE: tuple[re.Pattern[str], ...] = tuple(re.compile(p) for p in NON_ZONE_MARKERS)
WORD_CHAR_RE = re.compile(r"\w")


def _whole_word(text: str, start: int, end: int) -> str:
    """Расширяет найденный фрагмент до границ слова, чтобы показать его в объяснении."""
    while start > 0 and WORD_CHAR_RE.match(text[start - 1]):
        start -= 1
    while end < len(text) and WORD_CHAR_RE.match(text[end]):
        end += 1
    return text[start:end].strip()


def _collect_evidence(
    note: str, normalized: str, patterns: tuple[re.Pattern[str], ...]
) -> list[str]:
    """Возвращает слова исходного текста, на которых сработали шаблоны (без повторов)."""
    words: list[str] = []
    for pattern in patterns:
        match = pattern.search(normalized)
        if match:
            word = _whole_word(note, match.start(), match.end())
            if word not in words:
                words.append(word)
    return words


def keyword_zone_check(note: str) -> ZoneVerdict:
    """Запасной режим: проверка зон по словарю основ слов, работает без API-ключа.

    Скор зоны - число сработавших групп признаков. Основная зона - с максимальным
    скором, дополнительные попадают в ответ, если набрали не меньше
    SECONDARY_ZONE_RATIO от максимума. Уверенность растёт с числом признаков
    и снижается, если в тексте есть маркеры организационных событий.
    """
    normalized = normalize(note)
    evidence = {
        zone: _collect_evidence(note, normalized, patterns)
        for zone, patterns in COMPILED_LEXICON.items()
    }
    scores = {zone: len(words) for zone, words in evidence.items() if words}
    non_zone_words = _collect_evidence(note, normalized, COMPILED_NON_ZONE)

    if not scores:
        if non_zone_words:
            return ZoneVerdict(
                relevant=False,
                confidence=FALLBACK_NON_ZONE_CONFIDENCE,
                reason=(
                    "Не относится к отслеживаемым зонам: описано организационное событие "
                    f"без реакции ребёнка ({', '.join(non_zone_words)})"
                ),
                source=FALLBACK_SOURCE,
            )
        return ZoneVerdict(
            relevant=False,
            confidence=FALLBACK_NO_EVIDENCE_CONFIDENCE,
            reason="Признаков отслеживаемых зон не найдено, требуется уточнение",
            source=FALLBACK_SOURCE,
        )

    top_score = max(scores.values())
    zones = tuple(
        zone
        for zone, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if score >= top_score * SECONDARY_ZONE_RATIO
    )
    confidence = min(
        FALLBACK_BASE_CONFIDENCE + FALLBACK_CONFIDENCE_STEP * top_score, FALLBACK_MAX_CONFIDENCE
    )
    if non_zone_words:
        confidence -= FALLBACK_CONFIDENCE_STEP

    matched_words = list(dict.fromkeys(word for zone in zones for word in evidence[zone]))
    return ZoneVerdict(
        relevant=True,
        confidence=round(confidence, CONFIDENCE_PRECISION),
        reason=f"{format_zones(zones)} (признаки: {', '.join(matched_words)})",
        zones=zones,
        source=FALLBACK_SOURCE,
    )


def cross_check(note: str, verdict: ZoneVerdict) -> ZoneVerdict:
    """Постконтроль ответа LLM словарём признаков.

    Если модель считает, что наблюдение не относится к зонам, а словарь находит
    признаки зоны, уверенность снижается до DISAGREEMENT_CONFIDENCE и запись
    уходит на уточнение. Так спорное решение модели не приводит к пропуску
    важного наблюдения. Обратный случай (модель нашла зону, словарь нет)
    не штрафуется: модель понимает формулировки, которых нет в словаре.
    """
    if verdict.relevant:
        return verdict
    keyword_verdict = keyword_zone_check(note)
    if not keyword_verdict.relevant:
        return verdict
    hint = keyword_verdict.reason[0].lower() + keyword_verdict.reason[1:]
    return replace(
        verdict,
        confidence=min(verdict.confidence, DISAGREEMENT_CONFIDENCE),
        reason=f"{verdict.reason.rstrip('.')}. Требуется уточнение: по ключевым словам {hint}",
    )


def analyze_zone(
    note: str, use_llm: bool | None = None, model: ChatModel | None = None
) -> ZoneVerdict:
    """Проверяет наблюдение через LLM, а при недоступности модели - по ключевым словам.

    Args:
        note: текст наблюдения.
        use_llm: True - использовать LLM, False - только запасной режим,
            None - LLM, если задан GIGACHAT_CREDENTIALS.
        model: готовый клиент модели (например, заглушка в тестах).

    Returns:
        Полный результат проверки с указанием источника решения.
    """
    if use_llm is False:
        return keyword_zone_check(note)

    uses_default_model = model is None
    model = model or get_default_model()
    if model is None:
        if use_llm:
            logger.warning("GigaChat недоступен: нет ключа, SDK или соединения")
        return keyword_zone_check(note)

    try:
        return cross_check(note, llm_zone_check(note, model))
    except LLMUnavailableError as error:
        message = str(error)[:LOG_ERROR_MAX_LENGTH]
        logger.warning("GigaChat недоступен (%s), дальше работает запасной режим", message)
        if uses_default_model:
            disable_default_model(message)
        return keyword_zone_check(note)
    except (ValueError, TypeError) as error:
        logger.warning(
            "Ответ LLM не прошёл проверку (%s), для записи включён запасной режим",
            str(error)[:LOG_ERROR_MAX_LENGTH],
        )
        return keyword_zone_check(note)


def check_zone(note: str, use_llm: bool | None = None) -> tuple[bool, float, str]:
    """Определяет, относится ли наблюдение к отслеживаемым зонам, и объясняет вывод.

    Args:
        note: текст наблюдения.
        use_llm: режим работы, см. analyze_zone.

    Returns:
        Кортеж (относится ли к зонам, уверенность от 0.0 до 1.0, объяснение).
    """
    return analyze_zone(note, use_llm).as_tuple()
