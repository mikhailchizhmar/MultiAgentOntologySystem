"""
relation_extractor.py
──────────────────────
Агент 3: RelationExtractor

Вход:  state.entities (list[Entity] от EntityClassifier)
Выход: state.relations (list[Relation])

Один LLM-вызов на предложение со всеми валидными парами сразу —
вместо одного вызова на каждую пару. Это сокращает число запросов
в 10–30 раз на типичном финансовом документе.
"""

from __future__ import annotations

import re
import json
from functools import lru_cache
from itertools import combinations

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import PipelineState, Entity, Relation

# pymorphy3 для лемматизации — нужен чтобы находить сущности
# в косвенных падежах. Если библиотека недоступна — fallback на substring match.
try:
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()
except ImportError:
    _morph = None


@lru_cache(maxsize=20000)
def _lemma(word: str) -> str:
    """Кэшированная лемматизация одного слова."""
    if not _morph:
        return word.lower()
    return _morph.parse(word)[0].normal_form


def _lemmatize_text(text: str) -> str:
    """Лемматизирует все слова в тексте, склейка через пробел."""
    words = re.findall(r"\w+", text.lower())
    return " ".join(_lemma(w) for w in words)


# ─────────────────────────────────────────────────────────────────────────────
# МАТРИЦА ДОПУСТИМЫХ ПАР КЛАССОВ
# Фильтрует пары до вызова LLM — не тратим токены на заведомо пустые комбинации
# ─────────────────────────────────────────────────────────────────────────────

# Направленные допустимые отношения (subject_cls, object_cls) → list[relation].
# Это ключевое: направление субъект→объект ЗАФИКСИРОВАНО матрицей,
# LLM не выбирает направление — она выбирает только тип отношения
# из заранее ограниченного списка для данной направленной пары.
# Так уходят абсурды вроде "Банк :issuedBy: кредит" (правильно: "кредит :issuedBy: Банк").
DIRECTED_RELATIONS: dict[tuple[str, str], list[str]] = {
    ("FinancialProduct", "ProductAttribute"): ["hasAttribute"],
    ("FinancialProduct", "Actor"):            ["issuedBy", "regulatedBy"],
    ("FinancialProduct", "Process"):          ["involves"],
    ("FinancialProduct", "Condition"):        ["requires"],
    ("FinancialProduct", "LegalTerm"):        ["requires", "involves"],
    ("FinancialProduct", "FinancialProduct"): ["subClassOf"],
    ("ProductAttribute", "Metric"):           ["hasValue"],
    ("Process",          "Actor"):            ["involves"],
    ("LegalTerm",        "Actor"):            ["involves"],
}


def get_allowed_relations(cls1: str, cls2: str) -> tuple[str, str, list[str]] | None:
    """
    Для пары (e1, e2) ищет правильное направление в DIRECTED_RELATIONS.
    Возвращает (subj_cls, obj_cls, allowed) или None если пара невалидна.
    Если допустимо только обратное направление — меняет порядок.
    """
    if (cls1, cls2) in DIRECTED_RELATIONS:
        return (cls1, cls2, DIRECTED_RELATIONS[(cls1, cls2)])
    if (cls2, cls1) in DIRECTED_RELATIONS:
        return (cls2, cls1, DIRECTED_RELATIONS[(cls2, cls1)])
    return None

# ─────────────────────────────────────────────────────────────────────────────
# ПРОМПТЫ
# Ключевое отличие от предыдущей версии: принимаем СПИСОК пар за один вызов
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ты — эксперт по финансовым онтологиям.
Для каждой пары (СУБЪЕКТ → ОБЪЕКТ) выбери ОДНО отношение из ПРЕДЛОЖЕННОГО
для этой пары списка, или "no_relation" если в предложении нет такой связи.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Направление субъект→объект УЖЕ ЗАДАНО в паре. НЕ меняй его.
2. Выбирай ТОЛЬКО из списка allowed для каждой пары.
3. Если в предложении связь между субъектом и объектом отсутствует
   или не описана явно — ставь "no_relation".
4. Лучше "no_relation", чем угаданная связь.

ЗНАЧЕНИЯ ОТНОШЕНИЙ:
- hasAttribute  — субъект (продукт) имеет атрибут (характеристику)
- issuedBy      — субъект (продукт) выпускается / предоставляется актором (объект)
- regulatedBy   — субъект (продукт) регулируется актором (объект)
- requires      — субъект (продукт) требует условие или юр.термин (объект)
- involves      — субъект включает в себя процесс / вовлекает актора (объект)
- subClassOf    — субъект является подтипом объекта
- hasValue      — атрибут (субъект) имеет числовое значение (объект)

ФОРМАТ ОТВЕТА — строго JSON-массив, по одному элементу на каждую пару в порядке входа:
[
  {"relation": "hasAttribute" | "no_relation" | ..., "confidence": 0.0-1.0, "evidence": "<=80 символов"},
  ...
]"""

USER_PROMPT_TEMPLATE = """Предложение: «{sentence}»

Пары (субъект → объект) и допустимые отношения для каждой:
{pairs_block}

Верни JSON-массив длиной ровно {n} — по одному объекту на каждую пару."""


# ─────────────────────────────────────────────────────────────────────────────
# АГЕНТ
# ─────────────────────────────────────────────────────────────────────────────

class RelationExtractorAgent:

    CONFIDENCE_THRESHOLD = 0.55

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, state: PipelineState) -> PipelineState:
        state.log("RelationExtractor",
                  f"Сущностей на входе: {len(state.entities)}")

        if len(state.entities) < 2:
            state.log("RelationExtractor", "Меньше 2 сущностей — пропускаю")
            return state

        try:
            sentences   = self._split_sentences(state.text)
            relations   = []
            rel_counter = 0
            total_pairs = 0
            total_calls = 0

            for sent_idx, (sent_text, sent_start, sent_end) in enumerate(sentences):
                sent_ents = self._entities_in_sentence(
                    state.entities, sent_text
                )
                if len(sent_ents) < 2:
                    continue

                # Собираем направленные пары: для каждой пары классов
                # уже определено кто субъект, кто объект, и какие отношения допустимы.
                # Это убирает абсурды вида "Банк :issuedBy: кредит" — направление
                # фиксировано матрицей DIRECTED_RELATIONS, а не выбором LLM.
                directed_pairs: list[tuple[Entity, Entity, list[str]]] = []
                for e1, e2 in combinations(sent_ents, 2):
                    spec = get_allowed_relations(e1.cls, e2.cls)
                    if spec is None:
                        continue
                    subj_cls, obj_cls, allowed = spec
                    # Ставим в правильном порядке субъект → объект
                    if e1.cls == subj_cls:
                        directed_pairs.append((e1, e2, allowed))
                    else:
                        directed_pairs.append((e2, e1, allowed))

                if not directed_pairs:
                    continue

                # Защита от перегрузки контекста: предложения с >25 пар
                # дробим на батчи. Это и убирает "Unterminated string" на
                # длинных параграфах с десятками сущностей.
                BATCH_SIZE = 20
                batches = [
                    directed_pairs[i:i + BATCH_SIZE]
                    for i in range(0, len(directed_pairs), BATCH_SIZE)
                ]

                total_pairs += len(directed_pairs)

                for batch_idx, batch in enumerate(batches):
                    total_calls += 1
                    batch_label = (f"{len(batch)} пар"
                                   if len(batches) == 1
                                   else f"{len(batch)} пар (батч {batch_idx+1}/{len(batches)})")
                    state.log("RelationExtractor",
                              f"Предложение {sent_idx+1}/{len(sentences)}: "
                              f"{batch_label} → 1 LLM-вызов")

                    results = self._classify_pairs_batch(batch, sent_text)

                    for (subj, obj, allowed), result in zip(batch, results):
                        rel = result["relation"]
                        if rel == "no_relation":
                            continue
                        # Жёсткая фильтрация: тип отношения должен быть
                        # в списке разрешённых для этой направленной пары
                        if rel not in allowed:
                            continue
                        if result["confidence"] < self.CONFIDENCE_THRESHOLD:
                            continue
                        rel_counter += 1
                        relations.append(Relation(
                            id=f"r{rel_counter}",
                            subject_id=subj.id,
                            subject_text=subj.text,
                            relation=rel,
                            object_id=obj.id,
                            object_text=obj.text,
                            confidence=result["confidence"],
                            evidence=result.get("evidence", sent_text[:100]),
                        ))

            state.relations = relations
            by_type: dict[str, int] = {}
            for r in relations:
                by_type[r.relation] = by_type.get(r.relation, 0) + 1

            state.log("RelationExtractor",
                      f"Итого: {total_pairs} пар, {total_calls} LLM-вызовов, "
                      f"{len(relations)} отношений найдено")
            if by_type:
                state.log("RelationExtractor",
                          ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))

        except Exception as e:
            state.errors.append(f"RelationExtractor: {e}")
            state.log("RelationExtractor", f"ОШИБКА: {e}")

        return state

    # ── Приватные методы ─────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """
        Сплиттер устойчив к юридическим аббревиатурам и нумерации.
        Защищает точки внутри 'п.', 'пп.', 'ст.', 'руб.', '2.1.', 'млн.' и т.п.
        от ложной интерпретации как конец предложения.
        """
        # 1) Защищаем точки в нумерации (2.1, 2.1.3) и аббревиатурах
        protected = re.sub(r'(\d)\.(\d)', r'\1·\2', text)
        protected = re.sub(
            r'\b(п|пп|ст|стст|абз|ч|разд|гл|подп|руб|коп|млн|млрд|тыс|г|гг|кв|м|см|др|пр|т|т\.е|т\.п|т\.д)\.',
            lambda m: m.group(1) + '·',
            protected,
            flags=re.IGNORECASE,
        )

        # 2) Разбиваем только по настоящим концам предложений:
        #    точка/!/? + пробел + заглавная буква (или начало нумерации типа "2.")
        result: list[tuple[str, int, int]] = []
        offset = 0
        parts  = re.split(r'(?<=[.!?])\s+(?=[А-ЯA-ZЁ\d])', protected)

        for part in parts:
            # Восстанавливаем защищённые точки
            sent = part.replace('·', '.').strip()
            if not sent:
                offset += len(part) + 1
                continue
            # Находим реальную позицию в исходном тексте
            start = text.find(sent, offset) if sent in text[offset:] else offset
            if start == -1:
                start = offset
            end = start + len(sent)
            result.append((sent, start, end))
            offset = end

        return result

    def _entities_in_sentence(
        self,
        entities:  list[Entity],
        sent_text: str,
    ) -> list[Entity]:
        """
        Находит сущности в предложении с учётом словоизменения.

        Стратегия:
        1) Прямое substring-совпадение (быстро, ловит точные формы и Metric).
        2) Лемматизация: и сущности, и предложения приводятся к нормальной форме.
           Это ловит "процентная ставка" в тексте "процентной ставки".

        Без шага 2 после нормализации в TermExtractor сущности в косвенных
        падежах становятся невидимы — это была корневая причина relation F1=0.
        """
        sent_lower    = sent_text.lower()
        sent_lemmas   = _lemmatize_text(sent_text) if _morph else None
        result = []

        for e in entities:
            text_lower = e.text.lower()

            # 1) Прямое вхождение
            if text_lower in sent_lower:
                result.append(e)
                continue

            # 2) Лемматизированное вхождение
            if sent_lemmas is not None:
                ent_lemmas = _lemmatize_text(e.text)
                if ent_lemmas and ent_lemmas in sent_lemmas:
                    result.append(e)

        return result

    def _classify_pairs_batch(
        self,
        pairs:       list[tuple[Entity, Entity, list[str]]],
        sent_text:   str,
        max_retries: int = 3,
    ) -> list[dict]:
        """
        Один LLM-вызов для всех пар в предложении.
        Каждая пара — направленная (subj, obj, allowed_relations).
        LLM выбирает только из allowed_relations или ставит no_relation.
        """

        # Формат: «субъект» ({subj_cls}) → «объект» ({obj_cls})  allowed: [list]
        pairs_block = "\n".join(
            f"{i+1}. «{subj.text}» ({subj.cls}) → «{obj.text}» ({obj.cls})  allowed: {allowed + ['no_relation']}"
            for i, (subj, obj, allowed) in enumerate(pairs)
        )

        user_msg = USER_PROMPT_TEMPLATE.format(
            sentence=sent_text,
            pairs_block=pairs_block,
            n=len(pairs),
        )

        no_relation_fallback = [
            {"relation": "no_relation", "confidence": 0.0, "evidence": ""}
            for _ in pairs
        ]

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.llm.bind(max_tokens=4096).invoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_msg),
                ])

                raw = response.content.strip()
                if raw.startswith("```"):
                    raw = "\n".join(raw.split("\n")[1:]).rsplit("```", 1)[0]

                # Убираем невалидные escape-последовательности.
                # LLM копирует фрагменты документа в поле "evidence" и иногда
                # оставляет одиночный \ (например из "Центр-инвест" или № п/п).
                # Валидные JSON-escapes: \" \\ \/ \b \f \n \r \t \uXXXX — их не трогаем.
                raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)

                results = json.loads(raw)

                while len(results) < len(pairs):
                    results.append({"relation": "no_relation", "confidence": 0.0, "evidence": ""})

                return results[:len(pairs)]

            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                print(f"  [RelationExtractor] Попытка {attempt}/{max_retries}: {e}. "
                      f"Повтор через {wait}с...")

        print(f"  [RelationExtractor] Все попытки исчерпаны: {last_error}")
        return no_relation_fallback