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
import os
from itertools import combinations

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import PipelineState, Entity, Relation

# ─────────────────────────────────────────────────────────────────────────────
# МАТРИЦА ДОПУСТИМЫХ ПАР КЛАССОВ
# Фильтрует пары до вызова LLM — не тратим токены на заведомо пустые комбинации
# ─────────────────────────────────────────────────────────────────────────────

VALID_PAIRS: set[tuple[str, str]] = {
    ("FinancialProduct", "ProductAttribute"),
    ("FinancialProduct", "Actor"),
    ("FinancialProduct", "Process"),
    ("FinancialProduct", "Condition"),
    ("FinancialProduct", "LegalTerm"),
    ("FinancialProduct", "FinancialProduct"),
    ("ProductAttribute", "Metric"),
    ("Actor",            "FinancialProduct"),
    ("Actor",            "Process"),
    ("Process",          "Actor"),
}

# ─────────────────────────────────────────────────────────────────────────────
# ПРОМПТЫ
# Ключевое отличие от предыдущей версии: принимаем СПИСОК пар за один вызов
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ты — эксперт по финансовым онтологиям.
Для каждой пары сущностей определи семантическое отношение в контексте предложения.

ДОПУСТИМЫЕ ОТНОШЕНИЯ:
- hasAttribute  — продукт имеет атрибут
- issuedBy      — продукт выпускается / предоставляется актором
- requires      — продукт требует условие или актора
- involves      — продукт / актор участвует в процессе
- regulatedBy   — продукт регулируется актором
- hasValue      — атрибут имеет числовое значение
- subClassOf    — является подтипом
- appliesTo     — применяется к продукту
- no_relation   — нет значимого онтологического отношения

ПРАВИЛА:
1. Выбирай отношение строго по содержанию предложения.
2. Если неочевидно — ставь no_relation.
3. confidence 0.9+ только при явной лексической поддержке.

ФОРМАТ — строго JSON, массив длиной ровно столько элементов, сколько пар на входе:
[
  {"relation": "...", "confidence": 0.0-1.0, "evidence": "фраза из текста"},
  ...
]"""

USER_PROMPT_TEMPLATE = """Предложение: «{sentence}»

Пары сущностей:
{pairs_block}

Верни JSON-массив — по одному объекту на каждую пару в том же порядке."""


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

                # Собираем валидные пары для этого предложения
                valid_pairs = [
                    (e1, e2)
                    for e1, e2 in combinations(sent_ents, 2)
                    if (e1.cls, e2.cls) in VALID_PAIRS
                    or (e2.cls, e1.cls) in VALID_PAIRS
                ]
                if not valid_pairs:
                    continue

                total_pairs += len(valid_pairs)
                total_calls += 1

                state.log("RelationExtractor",
                          f"Предложение {sent_idx+1}/{len(sentences)}: "
                          f"{len(valid_pairs)} пар → 1 LLM-вызов")

                # Один вызов на всё предложение
                results = self._classify_pairs_batch(valid_pairs, sent_text)

                for (e1, e2), result in zip(valid_pairs, results):
                    if result["relation"] == "no_relation":
                        continue
                    if result["confidence"] < self.CONFIDENCE_THRESHOLD:
                        continue
                    rel_counter += 1
                    relations.append(Relation(
                        id=f"r{rel_counter}",
                        subject_id=e1.id,
                        subject_text=e1.text,
                        relation=result["relation"],
                        object_id=e2.id,
                        object_text=e2.text,
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
        return [e for e in entities if e.text.lower() in sent_text.lower()]

    def _classify_pairs_batch(
        self,
        pairs:     list[tuple[Entity, Entity]],
        sent_text: str,
    ) -> list[dict]:
        """
        Один LLM-вызов для всех пар в предложении.
        Возвращает список результатов в том же порядке, что и pairs.
        """
        pairs_block = "\n".join(
            f"{i+1}. «{e1.text}» ({e1.cls}) ↔ «{e2.text}» ({e2.cls})"
            for i, (e1, e2) in enumerate(pairs)
        )

        user_msg = USER_PROMPT_TEMPLATE.format(
            sentence=sent_text,
            pairs_block=pairs_block,
        )

        response = self.llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:]).rsplit("```", 1)[0]

        results = json.loads(raw)

        # Защита: если LLM вернул меньше элементов чем пар — дополняем no_relation
        while len(results) < len(pairs):
            results.append({"relation": "no_relation", "confidence": 0.0, "evidence": ""})

        return results[:len(pairs)]

