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

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import PipelineState, Term


# ─────────────────────────────────────────────────────────────────────────────
# ПРОМПТЫ
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ты — эксперт по финансовой терминологии и NLP.
Твоя задача — извлечь из текста финансового документа все термины и понятия,
которые имеют значение для построения онтологии финансовых продуктов.

ИЗВЛЕКАЙ:
- названия финансовых продуктов и инструментов
- атрибуты и характеристики продуктов (ставки, сроки, суммы, лимиты)
- участников и стороны (банк, заёмщик, регулятор)
- бизнес-процессы (погашение, начисление, страхование)
- условия и ограничения
- юридические термины, специфичные для финансового домена
- числовые метрики (конкретные значения: %, рубли, сроки)

НЕ ИЗВЛЕКАЙ:
- общеупотребительные слова без финансового смысла ("также", "является", "данный")
- предлоги, союзы, вводные слова

ФОРМАТ ОТВЕТА — строго JSON, без markdown:
{
  "terms": [
    {
      "text": "термин как в тексте",
      "context": "предложение из текста, где встретился термин",
      "confidence": 0.0-1.0,
      "reason": "одна строка — почему этот термин важен для онтологии"
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

def llm_extraction(
    text:        str,
    doc_type:    str,
    product:     str,
    candidates:  list[dict],
    llm:         ChatOpenAI,
) -> list[dict]:
    """
    GPT-4o верифицирует кандидатов и добавляет пропущенные термины.
    Возвращает список словарей из JSON-ответа модели.
    """
    # Форматируем статистических кандидатов для промпта
    cand_str = "\n".join(
        f"  - {c['text']} (score={c['score']})"
        for c in candidates[:15]  # топ-15 чтобы не раздувать контекст
    )

    user_msg = USER_PROMPT_TEMPLATE.format(
        doc_type=doc_type,
        product=product,
        text=text,
        statistical_candidates=cand_str or "  (нет кандидатов)",
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Убираем возможные markdown-обёртки
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        raw = raw.rsplit("```", 1)[0]

    data = json.loads(raw)
    return data.get("terms", [])


# ─────────────────────────────────────────────────────────────────────────────
# ОБЪЕДИНЕНИЕ РЕЗУЛЬТАТОВ
# ─────────────────────────────────────────────────────────────────────────────

def merge_terms(
    statistical: list[dict],
    llm_terms:   list[dict],
) -> list[Term]:
    """
    Объединяет результаты двух проходов.
    Дедуплицирует по нормализованному тексту.
    Термины из обоих источников получают бонус к score.
    """
    stat_texts = {t["text"].lower(): t for t in statistical}
    merged: dict[str, Term] = {}

    # Сначала статистические
    for t in statistical:
        key = t["text"].lower()
        merged[key] = Term(
            text=t["text"],
            context=t["context"],
            score=t["score"],
            source="statistical",
            frequency=1,
        )

    # Добавляем LLM-термины
    for t in llm_terms:
        key = t["text"].lower()
        if key in merged:
            # Термин найден обоими методами — повышаем score
            merged[key].source    = "both"
            merged[key].score     = min(merged[key].score * 1.3, 1.0)
            merged[key].frequency += 1
        else:
            merged[key] = Term(
                text=t["text"],
                context=t.get("context", ""),
                score=t.get("confidence", 0.7),
                source="llm",
                frequency=1,
            )

    # Сортируем: "both" > "llm" > "statistical", внутри — по score
    source_priority = {"both": 2, "llm": 1, "statistical": 0}
    result = sorted(
        merged.values(),
        key=lambda t: (source_priority[t.source], t.score),
        reverse=True,
    )

    return result


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
