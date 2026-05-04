"""
auto_annotate.py
────────────────
Автоматическая разметка корпуса финансовых документов через LLM.

Использование:
    python auto_annotate.py                     # разметить весь корпус
    python auto_annotate.py --doc doc_001       # разметить один документ
    python auto_annotate.py --doc doc_001 doc_005  # несколько документов

Требования:
    pip install openai python-dotenv

Переменные среды (файл .env):
    OPENAI_API_KEY=sk-...
    LLM_MODEL=gpt-4o          # или gpt-4o-mini для экономии
"""

import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

# ── Зависимости ──────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("pip install openai")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env не обязателен — можно передать ключ напрямую

# ── Импорт корпуса ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from corpus.documents import CORPUS

# ── Константы ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "annotations"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL = os.getenv("LLM_MODEL", "gpt-4o")
MAX_RETRIES = 3
RETRY_DELAY = 5  # секунд

# ─────────────────────────────────────────────────────────────────────────────
# ПРОМПТЫ
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ты — эксперт по финансовым онтологиям и NLP-разметке.
Твоя задача — размечать тексты финансовых документов для построения онтологии финансовых продуктов.

СХЕМА КЛАССОВ СУЩНОСТЕЙ:
- FinancialProduct   — финансовый продукт или инструмент (кредит, вклад, ПИФ, облигация, карта...)
- ProductAttribute   — атрибут/характеристика продукта (ставка, срок, сумма, тип платежа, лимит...)
- Actor              — участник/сторона (банк, заёмщик, УК, страховщик, регулятор, эмитент...)
- Process            — бизнес-процесс (погашение, выдача, начисление, страхование, листинг...)
- Condition          — условие или ограничение (требование к возрасту, минимальный остаток, срок владения...)
- LegalTerm          — юридический термин (договор, оферта, залог, поручительство, выгодоприобретатель...)
- Metric             — числовое значение или измеримый показатель (10.5% годовых, 1000 рублей, 20 лет...)

СХЕМА ОТНОШЕНИЙ:
- hasAttribute       — продукт имеет атрибут           (FinancialProduct → ProductAttribute)
- issuedBy           — продукт выпускается актором      (FinancialProduct → Actor)
- requires           — продукт/процесс требует условия  (FinancialProduct/Process → Condition/Actor)
- involves           — продукт/актор участвует в проц.  (FinancialProduct/Actor → Process)
- regulatedBy        — регулируется актором             (FinancialProduct → Actor)
- subClassOf         — является подтипом                (FinancialProduct → FinancialProduct)
- hasValue           — атрибут имеет значение           (ProductAttribute → Metric)
- appliesTo          — применяется к продукту           (любое → FinancialProduct)

ПРАВИЛА РАЗМЕТКИ:
1. Извлекай только сущности, явно присутствующие в тексте.
2. Числовые значения (%, суммы, сроки) размечай как Metric, а не как ProductAttribute.
3. Если термин неоднозначен — выбирай наиболее вероятный класс и снижай confidence.
4. confidence — твоя уверенность в разметке от 0.0 до 1.0.
5. Для каждого отношения указывай evidence — цитату из текста, подтверждающую отношение.

ФОРМАТ ОТВЕТА — строго JSON, без markdown-обёрток, без пояснений вне JSON:
{
  "entities": [
    {
      "id": "e1",
      "text": "...",
      "class": "FinancialProduct",
      "confidence": 0.95,
      "comment": "краткое пояснение"
    }
  ],
  "relations": [
    {
      "id": "r1",
      "subject": "e1",
      "relation": "hasAttribute",
      "object": "e2",
      "confidence": 0.9,
      "evidence": "цитата из текста"
    }
  ],
  "ontology_triples": [
    "КлассA :отношение :КлассB"
  ],
  "annotation_notes": "краткий комментарий о сложных случаях"
}"""

USER_PROMPT_TEMPLATE = """Размести следующий финансовый документ.

ID документа: {doc_id}
Тип документа: {doc_type}
Продукт: {doc_product}

ТЕКСТ:
{text}

Верни ТОЛЬКО JSON разметку."""


# ─────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def clean_json_response(raw: str) -> str:
    """Убирает возможные markdown-обёртки из ответа LLM."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        # убрать первую и последнюю строки (```json и ```)
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return raw.strip()


def annotate_document(client: OpenAI, doc: dict, verbose: bool = True) -> dict:
    """
    Отправляет один документ на разметку в LLM.
    Возвращает словарь с разметкой + метаданными.
    """
    doc_id = doc["id"]
    if verbose:
        print(f"  → Размечаю {doc_id} ({doc['product']})...")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        doc_id=doc_id,
        doc_type=doc["type"],
        doc_product=doc["product"],
        text=doc["text"]
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.0,   # детерминированность важна для разметки
                max_tokens=4096,
                response_format={"type": "json_object"}  # GPT-4o поддерживает
            )

            raw_text = response.choices[0].message.content
            cleaned  = clean_json_response(raw_text)
            annotation = json.loads(cleaned)

            # Добавляем метаданные
            result = {
                "id":          doc_id,
                "doc_type":    doc["type"],
                "product":     doc["product"],
                "source":      doc["source"],
                "text":        doc["text"],
                "annotator":   f"llm:{MODEL}",
                "timestamp":   datetime.utcnow().isoformat(),
                "annotation":  annotation,
                "status":      "ok"
            }

            if verbose:
                n_ent = len(annotation.get("entities", []))
                n_rel = len(annotation.get("relations", []))
                print(f"     ✓ {n_ent} сущностей, {n_rel} отношений")

            return result

        except json.JSONDecodeError as e:
            last_error = f"JSONDecodeError: {e}"
            if verbose:
                print(f"     ✗ Попытка {attempt}/{MAX_RETRIES}: {last_error}")
            time.sleep(RETRY_DELAY)

        except Exception as e:
            last_error = str(e)
            if verbose:
                print(f"     ✗ Попытка {attempt}/{MAX_RETRIES}: {last_error}")
            time.sleep(RETRY_DELAY)

    # Все попытки исчерпаны
    return {
        "id":       doc_id,
        "text":     doc["text"],
        "annotator": f"llm:{MODEL}",
        "timestamp": datetime.utcnow().isoformat(),
        "status":   "error",
        "error":    last_error
    }


def save_annotation(result: dict):
    """Сохраняет разметку одного документа в JSON."""
    path = OUTPUT_DIR / f"{result['id']}_annotation.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def save_all_annotations(results: list):
    """Сохраняет все разметки в один сводный файл."""
    path = OUTPUT_DIR / "all_annotations.json"
    summary = {
        "meta": {
            "model":     MODEL,
            "timestamp": datetime.utcnow().isoformat(),
            "total":     len(results),
            "ok":        sum(1 for r in results if r.get("status") == "ok"),
            "errors":    sum(1 for r in results if r.get("status") == "error"),
        },
        "documents": results
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ОСНОВНОЙ СКРИПТ
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Автоматическая разметка корпуса через LLM")
    parser.add_argument("--doc", nargs="*", help="ID документов для разметки (по умолчанию — все)")
    parser.add_argument("--api-key", help="OpenAI API key (или задай OPENAI_API_KEY в .env)")
    parser.add_argument("--quiet", action="store_true", help="Минимальный вывод")
    args = parser.parse_args()

    # Ключ API
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Не найден OpenAI API key.\n"
            "Задай переменную OPENAI_API_KEY или передай --api-key"
        )

    global MODEL
    MODEL = os.getenv("LLM_MODEL")
    if not MODEL:
            raise ValueError(
                "Не найден LLM_MODEL.\n"
                "Задай переменную LLM_MODEL в .env"
            )

    client = OpenAI(api_key=api_key)

    # Выбор документов
    if args.doc:
        docs_to_process = [d for d in CORPUS if d["id"] in args.doc]
        if not docs_to_process:
            print(f"Документы не найдены: {args.doc}")
            return
    else:
        docs_to_process = CORPUS

    verbose = not args.quiet
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Модель:     {MODEL}")
        print(f"  Документов: {len(docs_to_process)}")
        print(f"  Выходная папка: {OUTPUT_DIR}")
        print(f"{'='*60}\n")

    results = []
    for i, doc in enumerate(docs_to_process, 1):
        if verbose:
            print(f"[{i}/{len(docs_to_process)}] {doc['id']}")

        result = annotate_document(client, doc, verbose=verbose)
        results.append(result)
        save_annotation(result)  # сохраняем сразу — на случай прерывания

        # Пауза между запросами (rate limiting)
        if i < len(docs_to_process):
            time.sleep(1)

    # Сводный файл
    summary_path = save_all_annotations(results)

    if verbose:
        ok    = sum(1 for r in results if r.get("status") == "ok")
        err   = sum(1 for r in results if r.get("status") == "error")
        print(f"\n{'='*60}")
        print(f"  Готово: {ok} успешно, {err} с ошибками")
        print(f"  Результаты: {summary_path}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
