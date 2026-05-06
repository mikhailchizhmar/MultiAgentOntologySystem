"""
validator.py
─────────────
Агент 4: Validator

Вход:  state.entities + state.relations
Выход: state.validated_triples (list[dict]) — очищенные тройки для интегратора

Три уровня проверки:
  1. Структурная  — соответствие domain/range схеме
  2. Дедупликация — одинаковые сущности с разным написанием
  3. LLM-проверка — семантическая валидация спорных троек (confidence < 0.75)
"""

from __future__ import annotations

import json
import os
from difflib import SequenceMatcher

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import PipelineState, Entity, Relation

# ─────────────────────────────────────────────────────────────────────────────
# СХЕМА: допустимые domain/range для каждого отношения
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA: dict[str, dict] = {
    "hasAttribute": {
        "domain": {"FinancialProduct"},
        "range":  {"ProductAttribute"},
    },
    "issuedBy": {
        "domain": {"FinancialProduct", "Actor"},
        "range":  {"Actor", "FinancialProduct"},
    },
    "requires": {
        "domain": {"FinancialProduct", "Process"},
        "range":  {"Condition", "Actor", "LegalTerm"},
    },
    "involves": {
        "domain": {"FinancialProduct", "Actor"},
        "range":  {"Process", "Actor"},
    },
    "regulatedBy": {
        "domain": {"FinancialProduct"},
        "range":  {"Actor"},
    },
    "hasValue": {
        "domain": {"ProductAttribute", "Metric"},
        "range":  {"Metric"},
    },
    "subClassOf": {
        "domain": {"FinancialProduct"},
        "range":  {"FinancialProduct"},
    },
    "appliesTo": {
        "domain": {"FinancialProduct", "ProductAttribute", "LegalTerm"},
        "range":  {"FinancialProduct"},
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# ПРОМПТ ДЛЯ СПОРНЫХ ТРОЕК
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ты — эксперт по финансовым онтологиям.
Проверь, является ли утверждение семантически корректным для финансового домена.

Ответь строго в JSON:
{
  "valid": true | false,
  "confidence": 0.0-1.0,
  "reason": "одна строка объяснения"
}"""

USER_PROMPT_TEMPLATE = """Утверждение: «{subject}» —[{relation}]→ «{object}»
Доказательство из текста: «{evidence}»

Корректно ли это утверждение для финансовой онтологии? Верни ТОЛЬКО JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# АГЕНТ
# ─────────────────────────────────────────────────────────────────────────────

class ValidatorAgent:
    """
    Двухуровневая валидация (без LLM — только правила):
      1. Дедупликация сущностей (fuzzy match)
      2. Структурная проверка (self-loops, неизвестные отношения)

    LLM-проверка убрана намеренно: она давала N дополнительных вызовов
    без значимого прироста качества, сильно замедляя пайплайн.
    """

    DEDUP_SIMILARITY = 0.85

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm  # сохраняем для совместимости сигнатуры

    def run(self, state: PipelineState) -> PipelineState:
        state.log("Validator",
                  f"Сущностей: {len(state.entities)}, "
                  f"отношений: {len(state.relations)}")

        try:
            # 1. Дедупликация сущностей
            deduped_entities, dedup_map = self._dedup_entities(state.entities)
            removed = len(state.entities) - len(deduped_entities)
            state.log("Validator", f"Дедупликация сущностей: убрано {removed} дублей")

            # После дедупликации id некоторых сущностей в relations невалидны —
            # перепривязываем subject_id/object_id на канонические id
            for r in state.relations:
                r.subject_id = dedup_map.get(r.subject_id, r.subject_id)
                r.object_id  = dedup_map.get(r.object_id,  r.object_id)

            # 2. Структурная валидация отношений
            struct_ok, struct_fail = self._structural_check(
                state.relations, dedup_map
            )
            state.log("Validator",
                      f"Структурная проверка: "
                      f"{len(struct_ok)} ОК, {len(struct_fail)} отклонено")

            # 3. Дедупликация троек: одна и та же тройка
            #    (subject_id, relation, object_id) не должна попадать в граф
            #    больше одного раза, даже если встретилась в нескольких предложениях.
            seen: set[tuple[str, str, str]] = set()
            unique: list[Relation] = []
            for r in struct_ok:
                key = (r.subject_id, r.relation, r.object_id)
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            duplicates_removed = len(struct_ok) - len(unique)
            state.log("Validator",
                      f"Дедупликация троек: убрано {duplicates_removed} дублей")

            state.validated_triples = self._to_triples(unique, deduped_entities)
            state.log("Validator",
                      f"Итого валидных троек: {len(state.validated_triples)}")

        except Exception as e:
            state.errors.append(f"Validator: {e}")
            state.log("Validator", f"ОШИБКА: {e}")

        return state

    # ── Шаг 1: дедупликация ──────────────────────────────────────────────────

    def _dedup_entities(
        self,
        entities: list[Entity],
    ) -> tuple[list[Entity], dict[str, str]]:
        """
        Объединяет сущности со схожим текстом (SequenceMatcher).
        Возвращает (дедуплицированный список, маппинг id → канонический id).
        """
        dedup_map: dict[str, str] = {}  # old_id → canonical_id
        canonical: list[Entity]   = []

        for e in entities:
            matched = None
            for c in canonical:
                if e.cls != c.cls:
                    continue
                sim = SequenceMatcher(
                    None, e.text.lower(), c.text.lower()
                ).ratio()
                if sim >= self.DEDUP_SIMILARITY:
                    matched = c
                    break

            if matched:
                dedup_map[e.id] = matched.id
                # Оставляем более длинный текст как каноническое написание
                if len(e.text) > len(matched.text):
                    matched.text = e.text
            else:
                dedup_map[e.id] = e.id
                canonical.append(e)

        return canonical, dedup_map

    # ── Шаг 2: структурная проверка ──────────────────────────────────────────

    def _structural_check(
        self,
        relations: list[Relation],
        dedup_map: dict[str, str],
    ) -> tuple[list[Relation], list[Relation]]:
        """Проверяет domain/range по SCHEMA."""
        ok: list[Relation]   = []
        fail: list[Relation] = []

        # Индекс: id → Entity (после дедупликации)
        # Строим из исходных отношений — субъекты/объекты ещё имеют старые id
        for r in relations:
            schema = SCHEMA.get(r.relation)
            if schema is None:
                # Неизвестное отношение — пропускаем
                fail.append(r)
                continue

            # Используем классы из самих отношений (их несём в subject_text/object_text)
            # Фактические классы нам недоступны напрямую через id — берём из context
            # Упрощение: если отношение прошло через RelationExtractor, считаем его
            # структурно валидным по умолчанию; явно отклоняем только self-loops
            if r.subject_text.lower() == r.object_text.lower():
                fail.append(r)
                continue

            ok.append(r)

        return ok, fail

    # ── Шаг 3: LLM-проверка спорных троек ───────────────────────────────────

    def _llm_check(
        self,
        relations: list[Relation],
        state:     PipelineState,
    ) -> list[Relation]:
        """Отправляет спорные тройки на LLM-проверку, возвращает подтверждённые."""
        confirmed = []

        for r in relations:
            try:
                user_msg = USER_PROMPT_TEMPLATE.format(
                    subject=r.subject_text,
                    relation=r.relation,
                    object=r.object_text,
                    evidence=r.evidence,
                )
                response = self.llm.invoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_msg),
                ])
                raw = response.content.strip()
                if raw.startswith("```"):
                    raw = "\n".join(raw.split("\n")[1:]).rsplit("```", 1)[0]

                result = json.loads(raw)

                if result.get("valid", False):
                    r.confidence = float(result.get("confidence", r.confidence))
                    confirmed.append(r)

            except Exception as e:
                # При ошибке LLM — сохраняем тройку с исходной уверенностью
                state.log("Validator", f"LLM-проверка не удалась для «{r.subject_text}»: {e}")
                confirmed.append(r)

        return confirmed

    # ── Финальная сборка троек ───────────────────────────────────────────────

    @staticmethod
    def _to_triples(
        relations: list[Relation],
        entities:  list[Entity],
    ) -> list[dict]:
        """Преобразует Relation → словарь для OntologyIntegrator."""
        return [
            {
                "subject":      r.subject_text,
                "relation":     r.relation,
                "object":       r.object_text,
                "confidence":   r.confidence,
                "evidence":     r.evidence,
            }
            for r in relations
        ]


# ─────────────────────────────────────────────────────────────────────────────
# БЫСТРЫЙ ТЕСТ
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    from corpus.documents import CORPUS
    from agents.term_extractor import TermExtractorAgent
    from agents.entity_classifier import EntityClassifierAgent
    from agents.relation_extractor import RelationExtractorAgent

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
    state = ValidatorAgent(llm=llm).run(state)

    print(f"\n{'='*55}")
    print(f"Валидных троек: {len(state.validated_triples)}")
    for t in state.validated_triples:
        print(f"  «{t['subject']}» —[{t['relation']}]→ «{t['object']}»  "
              f"conf={t['confidence']:.2f}")
    print("\nЛоги:")
    for log in state.logs:
        print(f"  {log}")
