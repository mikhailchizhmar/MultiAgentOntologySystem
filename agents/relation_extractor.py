"""
relation_extractor.py
──────────────────────
Агент 3: RelationExtractor

Вход:  state.entities (list[Entity] от EntityClassifier)
Выход: state.relations (list[Relation])

Два прохода:
  1. Попарный анализ сущностей в одном предложении → LLM решает, есть ли отношение
  2. Структурные отношения (subClassOf) по типам продуктов → правила

gpt-4o-mini вызывается по одному предложению за раз — дёшево и точно.
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
# ДОПУСТИМЫЕ ОТНОШЕНИЯ ПО ПАРАМ КЛАССОВ
# Матрица фильтрует заведомо невалидные пары до вызова LLM
# ─────────────────────────────────────────────────────────────────────────────

VALID_PAIRS: set[tuple[str, str]] = {
    ("FinancialProduct", "ProductAttribute"),  # hasAttribute
    ("FinancialProduct", "Actor"),             # issuedBy / regulatedBy
    ("FinancialProduct", "Process"),           # involves
    ("FinancialProduct", "Condition"),         # requires
    ("FinancialProduct", "LegalTerm"),         # hasAttribute (обеспечение)
    ("FinancialProduct", "FinancialProduct"),  # subClassOf
    ("ProductAttribute", "Metric"),            # hasValue
    ("Actor",            "FinancialProduct"),  # issuedBy
    ("Actor",            "Process"),           # involves
    ("Process",          "Actor"),             # involves
}

RELATION_TYPES = [
    "hasAttribute",
    "issuedBy",
    "requires",
    "involves",
    "regulatedBy",
    "hasValue",
    "subClassOf",
    "appliesTo",
    "no_relation",   # специальный тип — отношения нет
]

# ─────────────────────────────────────────────────────────────────────────────
# ПРОМПТЫ
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ты — эксперт по финансовым онтологиям.
Определи семантическое отношение между двумя сущностями в контексте предложения.

ДОПУСТИМЫЕ ОТНОШЕНИЯ:
- hasAttribute  — продукт имеет атрибут (ипотечный кредит → процентная ставка)
- issuedBy      — продукт выпускается / предоставляется актором (кредит → банк)
- requires      — продукт требует условие (кредит → страхование залога)
- involves      — продукт / актор участвует в процессе (кредит → погашение)
- regulatedBy   — продукт регулируется актором (кредит → ЦБ РФ)
- hasValue      — атрибут имеет числовое значение (процентная ставка → 10,5%)
- subClassOf    — является подтипом (ипотечный кредит → кредитный продукт)
- appliesTo     — применяется к продукту (инвестиционный пай → ПИФ)
- no_relation   — между сущностями нет значимого онтологического отношения

ПРАВИЛА:
1. Выбирай отношение строго по содержанию предложения, не по умолчанию.
2. Если отношение неочевидно — ставь no_relation с confidence < 0.5.
3. confidence отражает уверенность: 0.9+ только при явной лексической поддержке.

ФОРМАТ — строго JSON:
{
  "relation": "одно из допустимых",
  "confidence": 0.0-1.0,
  "evidence": "ключевая фраза из предложения, подтверждающая отношение"
}"""

USER_PROMPT_TEMPLATE = """Предложение: «{sentence}»

Сущность 1: «{e1_text}» (класс: {e1_cls})
Сущность 2: «{e2_text}» (класс: {e2_cls})

Какое отношение между ними? Верни ТОЛЬКО JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# АГЕНТ
# ─────────────────────────────────────────────────────────────────────────────

class RelationExtractorAgent:
    """
    Извлекает отношения между сущностями в пределах одного предложения.
    Фильтрует заведомо невалидные пары через VALID_PAIRS до вызова LLM.
    """

    CONFIDENCE_THRESHOLD = 0.55  # ниже — отбрасываем

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, state: PipelineState) -> PipelineState:
        state.log("RelationExtractor",
                  f"Сущностей на входе: {len(state.entities)}")

        if len(state.entities) < 2:
            state.log("RelationExtractor", "Меньше 2 сущностей — пропускаю")
            return state

        try:
            # Индекс сущностей по позиции в тексте
            ent_map = {e.id: e for e in state.entities}

            # Разбиваем текст на предложения
            sentences = self._split_sentences(state.text)

            relations  = []
            rel_counter = 0

            for sent_text, sent_start, sent_end in sentences:
                # Сущности в этом предложении
                sent_ents = self._entities_in_sentence(
                    state.entities, state.text, sent_text, sent_start, sent_end
                )
                if len(sent_ents) < 2:
                    continue

                # Перебираем все пары
                for e1, e2 in combinations(sent_ents, 2):
                    # Фильтр по матрице допустимых пар
                    if (e1.cls, e2.cls) not in VALID_PAIRS and \
                       (e2.cls, e1.cls) not in VALID_PAIRS:
                        continue

                    result = self._classify_pair(e1, e2, sent_text)

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
            state.log("RelationExtractor",
                      f"Найдено отношений: {len(relations)}")

            # Краткая статистика по типам
            by_type: dict[str, int] = {}
            for r in relations:
                by_type[r.relation] = by_type.get(r.relation, 0) + 1
            state.log("RelationExtractor",
                      ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))

        except Exception as e:
            state.errors.append(f"RelationExtractor: {e}")
            state.log("RelationExtractor", f"ОШИБКА: {e}")

        return state

    # ── Приватные методы ─────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Возвращает (текст_предложения, start, end)."""
        result = []
        for m in re.finditer(r'[^.!?]+[.!?]?', text):
            s = m.group().strip()
            if s:
                result.append((s, m.start(), m.end()))
        return result

    def _entities_in_sentence(
        self,
        entities:   list[Entity],
        full_text:  str,
        sent_text:  str,
        sent_start: int,
        sent_end:   int,
    ) -> list[Entity]:
        """Возвращает сущности, упомянутые в данном предложении."""
        result = []
        for e in entities:
            # Ищем текст сущности в предложении (без учёта регистра)
            if e.text.lower() in sent_text.lower():
                result.append(e)
        return result

    def _classify_pair(
        self,
        e1:   Entity,
        e2:   Entity,
        sent: str,
    ) -> dict:
        """Вызывает LLM для одной пары сущностей."""
        user_msg = USER_PROMPT_TEMPLATE.format(
            sentence=sent,
            e1_text=e1.text,
            e1_cls=e1.cls,
            e2_text=e2.text,
            e2_cls=e2.cls,
        )

        response = self.llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:]).rsplit("```", 1)[0]

        return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# БЫСТРЫЙ ТЕСТ
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    from corpus.documents import CORPUS
    from agents.term_extractor import TermExtractorAgent
    from agents.entity_classifier import EntityClassifierAgent

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    doc = next(d for d in CORPUS if d["id"] == "doc_001")
    state = PipelineState(
        doc_id=doc["id"], doc_type=doc["type"],
        product=doc["product"], text=doc["text"],
    )

    corpus_texts = [d["text"] for d in CORPUS]
    state = TermExtractorAgent(llm=llm, corpus_texts=corpus_texts).run(state)
    state = EntityClassifierAgent(llm=llm).run(state)
    state = RelationExtractorAgent(llm=llm).run(state)

    print(f"\n{'='*55}")
    for r in state.relations:
        print(f"  «{r.subject_text}» —[{r.relation}]→ «{r.object_text}»  "
              f"conf={r.confidence:.2f}")
        print(f"    ↳ {r.evidence[:80]}")
    print("\nЛоги:")
    for log in state.logs:
        print(f"  {log}")
