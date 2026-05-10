"""
term_extractor.py
──────────────────
Агент 1: TermExtractor

Задача: принять текст документа, вернуть список финансовых терминов-кандидатов.

Два прохода:
  1. Статистический — TF-IDF-подобный скоринг n-грамм по корпусу.
     Быстро, дёшево, без API-вызовов. Даёт «скелет» терминологии.

  2. LLM — GPT-4o верифицирует и дополняет список.
     Ловит то, что статистика пропустила: многословные термины,
     аббревиатуры, контекстно-зависимые понятия.

Выход: list[Term] — передаётся в EntityClassifier.

Зависимости:
    pip install openai langchain-openai
"""

from __future__ import annotations

import re
import json
import math
import os
from collections import Counter
from typing import Any

# pymorphy3 используется ТОЛЬКО для лемматизации в relation_extractor.
# Нормализацию падежей терминов делает LLM через промпт — это надёжнее
# чем поморфный разбор, который ломает согласование в составных терминах
# («3 рабочих дня» → «3 рабочий день», «100 000 рублей» → «100 000 рубль»).

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import PipelineState, Term


# ─────────────────────────────────────────────────────────────────────────────
# ПРОМПТЫ
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ты — эксперт по построению онтологий финансовых продуктов.
Извлеки из документа все термины, имеющие онтологическое значение,
независимо от типа документа (кредитный, депозитный, инвестиционный, страховой,
карточный, нормативный).

ЧТО СЧИТАТЬ ТЕРМИНОМ — пройди документ и ищи 7 типов сущностей:

1. ПРОДУКТ — то, что предлагается клиенту: кредит, вклад, ПИФ, облигация,
   карта, полис, счёт. Извлекай и обобщённое название («вклад»),
   и конкретный подвид («срочный вклад с пополнением», «зарплатный овердрафт»).

2. УЧАСТНИК — любая сторона или организация: клиент, банк, эмитент, регулятор,
   управляющая компания, страховщик, поручитель, выгодоприобретатель,
   государственный орган, биржа, депозитарий.

3. АТРИБУТ — характеристика продукта БЕЗ числового значения:
   ставка, срок, сумма, лимит, валюта, способ начисления, схема платежей,
   условия пополнения/снятия, тип ставки, базис расчёта, тарифный план.

4. ПРОЦЕСС — действие или операция, связанные с продуктом:
   выдача, погашение, начисление, конвертация, размещение, снятие,
   расторжение, переуступка, реструктуризация, страховое возмещение.

5. УСЛОВИЕ — ограничение или требование при котором что-то применяется:
   развёрнутое условие извлекай ЦЕЛИКОМ как одну сущность
   («при сроке вклада свыше 6 месяцев», «при отсутствии просрочки»).

6. ЮРИДИЧЕСКИЙ ТЕРМИН — правовая конструкция: залог, поручительство, оферта,
   акцепт, неустойка, цессия, персональные данные, страховой случай.

7. МЕТРИКА — конкретное числовое значение с единицей измерения:
   проценты, суммы в любой валюте, сроки в днях/месяцах/годах, количества,
   относительные величины («полуторакратный размер»).
   Извлекай каждое значение как ОТДЕЛЬНУЮ сущность, не сливай с атрибутом.

ПРАВИЛА ИЗВЛЕЧЕНИЯ:

- Извлекай термин один раз, даже если он встречается в тексте многократно.
- Многословные термины оставляй целиком — не разбивай на части.
  «бюро кредитных историй» — одна сущность, не три.
- Развёрнутые условия и формулировки оставляй как есть, не упрощай:
  «более 60 календарных дней просрочки в течение 180 дней» — это одна сущность.
- Если в тексте есть и общее понятие («кредит»), и его подвид
  («ипотечный кредит»), извлекай оба — они образуют иерархию.

НОРМАЛИЗАЦИЯ ПАДЕЖА:
Приводи термин в именительный падеж, сохраняя грамматическое согласование
(прилагательное согласуется с существительным, числительное управляет падежом).
Если в тексте «процентной ставки» — возвращай «процентная ставка».
Если «3 рабочих дней» — возвращай «3 рабочих дня» (числительное 3 требует
родительного падежа существительного — это согласование, а не ошибка).
Многословные термины нормализуй целиком, не по словам.

НЕ ИЗВЛЕКАЙ:
- Общеупотребительные слова без доменного смысла: «также», «является»,
  «настоящий», «данный», «соответствующий».
- Усечённые формы и фрагменты: «процент» отдельно от «процентная ставка»,
  «индивидуальный» отдельно от «индивидуальные условия».
- Артефакты форматирования: одиночные предлоги, союзы, номера пунктов.

ФОРМАТ ОТВЕТА — строго JSON, без markdown:
{
  "terms": [
    {
      "text": "термин в именительном падеже",
      "context": "предложение из текста, где встретился термин",
      "confidence": 0.0-1.0,
      "reason": "одна строка — какой это тип (продукт/участник/атрибут/процесс/условие/юр.термин/метрика) и почему важен"
    }
  ]
}"""

USER_PROMPT_TEMPLATE = """Извлеки финансовые термины из следующего документа.

Тип документа: {doc_type}
Продукт: {product}

ТЕКСТ:
{text}

Статистически найденные кандидаты (можешь использовать как подсказку, но не ограничивайся ими):
{statistical_candidates}

Верни ТОЛЬКО JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# СТАТИСТИЧЕСКИЙ ЭКСТРАКТОР (проход 1)
# ─────────────────────────────────────────────────────────────────────────────

# Стоп-слова для русского языка (минимальный набор)
STOPWORDS = {
    "и", "в", "на", "с", "по", "для", "не", "от", "до", "из", "к",
    "о", "об", "при", "за", "как", "что", "это", "но", "а", "или",
    "же", "бы", "ли", "то", "так", "если", "все", "уже", "ещё",
    "более", "менее", "также", "который", "которая", "которое",
    "которые", "является", "являются", "составляет", "составляют",
    "данный", "данная", "данные", "каждый", "любой", "весь",
    "без", "под", "над", "между", "через", "после", "перед",
}


def extract_ngrams(text: str, n: int) -> list[str]:
    """Извлекает n-граммы из текста."""
    # Токенизация: только слова (не знаки препинания)
    tokens = re.findall(r'[а-яёА-ЯЁa-zA-Z]+', text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def statistical_extraction(
    text: str,
    corpus_texts: list[str] | None = None,
    top_k: int = 20,
) -> list[dict]:
    """
    TF-IDF-подобный скоринг n-грамм (1..3).

    TF  = частота n-граммы в документе
    IDF = log(N / df), где N — размер корпуса, df — число документов с термином
    Если корпус не передан — используем только TF (упрощённый режим).
    """
    # Собираем все n-граммы документа (1, 2, 3)
    all_ngrams: list[str] = []
    for n in (1, 2, 3):
        all_ngrams.extend(extract_ngrams(text, n))

    if not all_ngrams:
        return []

    tf = Counter(all_ngrams)
    total = sum(tf.values())

    # IDF по корпусу
    if corpus_texts:
        N = len(corpus_texts)
        df = Counter()
        for doc in corpus_texts:
            doc_ngrams = set()
            for n in (1, 2, 3):
                doc_ngrams.update(extract_ngrams(doc, n))
            for ng in doc_ngrams:
                df[ng] += 1
        scores = {
            ng: (count / total) * math.log((N + 1) / (df.get(ng, 0) + 1))
            for ng, count in tf.items()
        }
    else:
        # Без корпуса: предпочитаем длинные n-граммы (они специфичнее)
        scores = {
            ng: (count / total) * (1 + 0.3 * (ng.count(' ')))
            for ng, count in tf.items()
        }

    # Фильтрация: убираем слишком короткие и стоп-слова
    scores = {
        ng: s for ng, s in scores.items()
        if len(ng) > 3 and not all(w in STOPWORDS for w in ng.split())
    }

    # Top-K
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Привязываем каждый термин к предложению-контексту
    sentences = re.split(r'(?<=[.!?])\s+', text)
    results = []
    for ngram, score in top:
        context = next(
            (s for s in sentences if ngram.lower() in s.lower()),
            text[:100]
        )
        results.append({
            "text":    ngram,
            "score":   round(score, 4),
            "context": context.strip(),
            "source":  "statistical",
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# LLM ЭКСТРАКТОР (проход 2)
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_chars: int = 12000) -> list[str]:
    """
    Режет длинный текст на чанки по границам предложений.
    Нужно потому что:
      1) LLM с большим текстом возвращает обрезанный JSON (Unterminated string).
      2) Качество извлечения деградирует на текстах > 10–15 KB.
    """
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chars and current:
            chunks.append(current)
            current = sent
        else:
            current = (current + " " + sent) if current else sent
    if current:
        chunks.append(current)
    return chunks


def llm_extraction(
    text:        str,
    doc_type:    str,
    product:     str,
    candidates:  list[dict],
    llm:         ChatOpenAI,
) -> list[dict]:
    """
    GPT-4o-mini верифицирует кандидатов и добавляет пропущенные термины.

    Длинные документы режутся на чанки — иначе ответ модели обрывается
    посередине строки JSON (ошибка "Unterminated string").
    Результаты по чанкам объединяются.

    max_tokens=4096 явно задан, чтобы ответ гарантированно помещался.
    """
    chunks = _chunk_text(text, max_chars=12000)
    cand_str = "\n".join(
        f"  - {c['text']} (score={c['score']})"
        for c in candidates[:15]
    ) or "  (нет кандидатов)"

    all_terms: list[dict] = []
    for chunk in chunks:
        user_msg = USER_PROMPT_TEMPLATE.format(
            doc_type=doc_type,
            product=product,
            text=chunk,
            statistical_candidates=cand_str,
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        # max_tokens=4096 + bind: на gpt-4o-mini это потолок ответа,
        # его хватает на ~80–120 терминов в JSON.
        response = llm.bind(max_tokens=4096).invoke(messages)
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            raw = raw.rsplit("```", 1)[0]

        try:
            data = json.loads(raw)
            all_terms.extend(data.get("terms", []))
        except json.JSONDecodeError:
            # Если конкретный чанк всё-таки сломался — пропускаем его,
            # остальные чанки по-прежнему обрабатываются.
            continue

    return all_terms


# ─────────────────────────────────────────────────────────────────────────────
# ОБЪЕДИНЕНИЕ РЕЗУЛЬТАТОВ
# ─────────────────────────────────────────────────────────────────────────────

def merge_terms(
    statistical: list[dict],
    llm_terms:   list[dict],
) -> list[Term]:
    """
    Объединяет результаты двух проходов.
    Дедуплицирует по lower() тексту.
    Нормализацию падежей делает LLM в SYSTEM_PROMPT —
    не трогаем тексты здесь, чтобы не сломать согласование
    составных терминов («3 рабочих дня», «100 000 рублей»).
    """
    merged: dict[str, Term] = {}

    def _add(text: str, context: str, score: float, source_type: str):
        key = text.lower().strip()
        if not key:
            return
        if key in merged:
            existing = merged[key]
            if source_type == "llm" and existing.source == "statistical":
                existing.source = "both"
                existing.score  = min(existing.score * 1.3, 1.0)
            existing.frequency += 1
        else:
            merged[key] = Term(
                text=text,
                context=context,
                score=score,
                source=source_type,
                frequency=1,
            )

    for t in statistical:
        _add(t["text"], t["context"], t["score"], "statistical")

    for t in llm_terms:
        _add(t["text"], t.get("context", ""), t.get("confidence", 0.7), "llm")

    source_priority = {"both": 2, "llm": 1, "statistical": 0}
    return sorted(
        merged.values(),
        key=lambda t: (source_priority[t.source], t.score),
        reverse=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# АГЕНТ
# ─────────────────────────────────────────────────────────────────────────────

class TermExtractorAgent:
    """
    Агент извлечения терминов.

    Используется как узел LangGraph:
        graph.add_node("extract_terms", TermExtractorAgent(llm).run)

    Также можно вызвать напрямую для тестирования:
        agent = TermExtractorAgent(llm)
        state = agent.run(state)
    """

    def __init__(
        self,
        llm:          ChatOpenAI,
        corpus_texts: list[str] | None = None,
        top_k_stat:   int = 20,
    ):
        self.llm          = llm
        self.corpus_texts = corpus_texts  # для IDF; если None — TF-only
        self.top_k_stat   = top_k_stat

    def run(self, state: PipelineState) -> PipelineState:
        """Точка входа для LangGraph узла."""
        state.log("TermExtractor", f"Начинаю обработку: {state.doc_id}")

        try:
            # Проход 1: статистика
            stat_candidates = statistical_extraction(
                text=state.text,
                corpus_texts=self.corpus_texts,
                top_k=self.top_k_stat,
            )
            state.log(
                "TermExtractor",
                f"Статистика: {len(stat_candidates)} кандидатов"
            )

            # Проход 2: LLM
            llm_terms = llm_extraction(
                text=state.text,
                doc_type=state.doc_type,
                product=state.product,
                candidates=stat_candidates,
                llm=self.llm,
            )
            state.log(
                "TermExtractor",
                f"LLM: {len(llm_terms)} терминов"
            )

            # Объединение
            merged = merge_terms(stat_candidates, llm_terms)
            state.terms = merged
            state.log(
                "TermExtractor",
                f"Итого после merge: {len(merged)} уникальных терминов "
                f"(both={sum(1 for t in merged if t.source=='both')}, "
                f"llm={sum(1 for t in merged if t.source=='llm')}, "
                f"stat={sum(1 for t in merged if t.source=='statistical')})"
            )

        except Exception as e:
            state.errors.append(f"TermExtractor: {e}")
            state.log("TermExtractor", f"ОШИБКА: {e}")

        return state


# ─────────────────────────────────────────────────────────────────────────────
# БЫСТРЫЙ ТЕСТ (запуск напрямую)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Тестируем на doc_001 (ипотека)
    from corpus.documents import CORPUS
    doc = next(d for d in CORPUS if d["id"] == "doc_001")

    state = PipelineState(
        doc_id=doc["id"],
        doc_type=doc["type"],
        product=doc["product"],
        text=doc["text"],
    )

    # Передаём все тексты корпуса для IDF
    corpus_texts = [d["text"] for d in CORPUS]

    agent = TermExtractorAgent(llm=llm, corpus_texts=corpus_texts)
    state = agent.run(state)

    print(f"\n{'='*55}")
    print(f"Документ: {state.doc_id} | {state.product}")
    print(f"Терминов найдено: {len(state.terms)}")
    print(f"{'='*55}")

    for t in state.terms:
        print(f"  [{t.source:12s}] score={t.score:.3f}  «{t.text}»")
        print(f"    контекст: {t.context[:80]}...")

    print("\nЛоги:")
    for log in state.logs:
        print(f"  {log}")

    if state.errors:
        print("\nОшибки:")
        for err in state.errors:
            print(f"  {err}")
