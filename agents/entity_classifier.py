"""
entity_classifier.py
─────────────────────
Агент 2: EntityClassifier

Вход:  state.terms  (list[Term]  от TermExtractor)
Выход: state.entities (list[Entity])

Стратегия — few-shot промпт с примерами из gold standard.
Батчинг: классифицируем по N терминов за один вызов, чтобы
экономить токены при использовании gpt-4o-mini.

Зависимости:
    pip install langchain langchain-openai
"""

from __future__ import annotations

import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import PipelineState, Term, Entity

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT ПРИМЕРЫ (из gold standard разметки)
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """ПРИМЕРЫ ПРАВИЛЬНОЙ КЛАССИФИКАЦИИ:

термин: "ипотечный кредит"
контекст: "Банк предоставляет заёмщику ипотечный кредит в размере 5 000 000 рублей"
класс: FinancialProduct
причина: конкретный вид кредитного продукта

термин: "процентная ставка"
контекст: "под процентную ставку 10,5% годовых"
класс: ProductAttribute
причина: характеристика продукта, не само числовое значение

термин: "10,5% годовых"
контекст: "под процентную ставку 10,5% годовых"
класс: Metric
причина: конкретное числовое значение атрибута

термин: "заёмщик"
контекст: "Банк предоставляет заёмщику ипотечный кредит"
класс: Actor
причина: сторона договора, получатель продукта

термин: "досрочное погашение"
контекст: "Досрочное погашение допускается без штрафных санкций"
класс: Process
причина: бизнес-процесс, связанный с продуктом

термин: "при владении менее 180 дней"
контекст: "Скидка при погашении паёв — не более 1% при владении менее 180 дней"
класс: Condition
причина: условие применения — не атрибут и не метрика

термин: "180 дней"
контекст: "при владении менее 180 дней"
класс: Condition
причина: в данном контексте это условие-порог, а не просто числовое значение

термин: "залог"
контекст: "Обеспечением по кредиту является залог приобретаемой недвижимости"
класс: LegalTerm
причина: юридическая конструкция обеспечения

термин: "управляющая компания"
контекст: "Управляющая компания осуществляет доверительное управление"
класс: Actor
причина: институциональный участник, управляющий фондом

термин: "стоимость чистых активов"
контекст: "Стоимость пая рассчитывается на основании стоимости чистых активов (СЧА)"
класс: ProductAttribute
причина: расчётный показатель фонда — атрибут, не конкретная цифра"""

SYSTEM_PROMPT = f"""Ты — эксперт по финансовым онтологиям.
Классифицируй каждый термин в один из классов:

КЛАССЫ:
- FinancialProduct  — финансовый продукт или инструмент
- ProductAttribute  — атрибут/характеристика продукта (не числовое значение!)
- Actor             — участник, сторона, организация
- Process           — бизнес-процесс
- Condition         — условие или ограничение
- LegalTerm         — юридический термин
- Metric            — числовое значение (%, рубли, сроки как числа)

КЛЮЧЕВЫЕ ПРАВИЛА:
1. Metric — только конкретные числа с единицами: "10,5% годовых", "5 000 000 рублей", "20 лет"
2. ProductAttribute — название характеристики без числа: "процентная ставка", "срок кредита"
3. Condition — срок/число как условие применения ("при владении менее 180 дней")
4. Если термин неоднозначен — снизь confidence, объясни в reason

{FEW_SHOT_EXAMPLES}

ФОРМАТ ОТВЕТА — строго JSON, без markdown:
{{
  "entities": [
    {{
      "text": "термин из входного списка",
      "class": "один из 7 классов",
      "confidence": 0.0-1.0,
      "comment": "одна строка объяснения"
    }}
  ]
}}"""

USER_PROMPT_TEMPLATE = """Классифицируй следующие термины.
Тип документа: {doc_type} | Продукт: {product}

ТЕРМИНЫ:
{terms_block}

Верни ТОЛЬКО JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# АГЕНТ
# ─────────────────────────────────────────────────────────────────────────────

class EntityClassifierAgent:
    """
    Классифицирует термины из state.terms → state.entities.
    Батчинг: BATCH_SIZE терминов за один LLM-вызов.
    """

    BATCH_SIZE = 15  # gpt-4o-mini справляется с 15 без деградации

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, state: PipelineState) -> PipelineState:
        state.log("EntityClassifier", f"Терминов на входе: {len(state.terms)}")

        if not state.terms:
            state.log("EntityClassifier", "Нет терминов — пропускаю")
            return state

        try:
            entities = []
            batches  = self._make_batches(state.terms)

            for i, batch in enumerate(batches):
                state.log("EntityClassifier",
                          f"Батч {i+1}/{len(batches)}: {len(batch)} терминов")
                classified = self._classify_batch(
                    batch, state.doc_type, state.product
                )
                entities.extend(classified)

            # Нумеруем сущности
            for idx, e in enumerate(entities, 1):
                e.id = f"e{idx}"

            state.entities = entities
            state.log("EntityClassifier",
                      f"Готово: {len(entities)} сущностей. "
                      + self._class_summary(entities))

        except Exception as e:
            state.errors.append(f"EntityClassifier: {e}")
            state.log("EntityClassifier", f"ОШИБКА: {e}")

        return state

    # ── Приватные методы ─────────────────────────────────────────────────────

    def _make_batches(self, terms: list[Term]) -> list[list[Term]]:
        return [
            terms[i:i + self.BATCH_SIZE]
            for i in range(0, len(terms), self.BATCH_SIZE)
        ]

    def _classify_batch(
        self,
        batch:    list[Term],
        doc_type: str,
        product:  str,
    ) -> list[Entity]:
        # Формируем блок терминов для промпта
        terms_block = "\n".join(
            f'{i+1}. термин: "{t.text}"\n   контекст: "{t.context[:120]}"'
            for i, t in enumerate(batch)
        )

        user_msg = USER_PROMPT_TEMPLATE.format(
            doc_type=doc_type,
            product=product,
            terms_block=terms_block,
        )

        response = self.llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:]).rsplit("```", 1)[0]

        data     = json.loads(raw)
        raw_ents = data.get("entities", [])

        # Собираем индекс term.text → Term для обогащения контекстом
        term_index = {t.text.lower(): t for t in batch}

        entities = []
        for item in raw_ents:
            text   = item.get("text", "")
            source = term_index.get(text.lower())
            entities.append(Entity(
                id="",   # проставим позже
                text=text,
                cls=item.get("class", "ProductAttribute"),
                context=source.context if source else "",
                confidence=float(item.get("confidence", 0.7)),
                comment=item.get("comment", ""),
            ))

        return entities

    @staticmethod
    def _class_summary(entities: list[Entity]) -> str:
        counts: dict[str, int] = {}
        for e in entities:
            counts[e.cls] = counts.get(e.cls, 0) + 1
        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


# ─────────────────────────────────────────────────────────────────────────────
# БЫСТРЫЙ ТЕСТ
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # import sys
    # sys.path.insert(0, "..")

    from corpus.documents import CORPUS
    from agents.term_extractor import TermExtractorAgent

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

    # Прогоняем через оба агента
    state = TermExtractorAgent(llm=llm, corpus_texts=[d["text"] for d in CORPUS]).run(state)
    state = EntityClassifierAgent(llm=llm).run(state)

    print(f"\n{'='*55}")
    for e in state.entities:
        print(f"  [{e.cls:<20}] conf={e.confidence:.2f}  «{e.text}»")
        if e.comment:
            print(f"    ↳ {e.comment}")

    print("\nЛоги:")
    for log in state.logs:
        print(f"  {log}")
